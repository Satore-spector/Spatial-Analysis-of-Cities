# Pull GHSL built-up surface (GHS-BUILT-S, 10 m, 2018) from Google Earth Engine
# and cut out the capital city of each southern Indian state, using the city
# boundaries that data_extraction_osm.R already wrote to data/<slug>_boundary.gpkg.
#
# Why 10 m / 2018: in GHSL release P2023A the 10 m built-up surface only exists
# for 2018 (100 m and 1 km go 1975-2030). 10 m is the finest resolution offered.
#
# Why only the cities and not whole states: a state at 10 m is billions of
# pixels (Karnataka alone is ~1.9 billion). Only the area inside each city
# boundary is downloaded.
#
# Setup (once):
#   install.packages(c("reticulate", "sf", "terra"))
#   The Python package earthengine-api is installed automatically by a recent
#   reticulate. (With an old reticulate, install Python 3.12 and run
#   reticulate::py_install("earthengine-api").)
#   You also need a Google Cloud project with the Earth Engine API enabled and
#   registered for Earth Engine use (https://code.earthengine.google.com/register).
#   The first run opens a browser to log in.
#
# Usage (from the project root):
#   Rscript data_extraction/ghsl_builtup_capitals.R --project=my-gcp-project
#   Rscript data_extraction/ghsl_builtup_capitals.R --project=my-gcp-project "Pune, India"
#
# Options:
#   --project=ID        Google Cloud project for Earth Engine (or set env var EE_PROJECT)
#   --extract-missing   run data_extraction_osm.R first for cities with no boundary file
#                       (note: that downloads POIs, buildings and roads too)
#   --force             rebuild outputs that already exist (downloaded tiles are reused;
#                       delete data/ghsl/tiles/<slug>/ to download them again)
#   --tile-deg=0.1      size of each download tile in degrees (smaller = more requests)
#   --summary=NAME.csv  name of the summary file in data/ghsl/ (default
#                       ghsl_capitals_summary.csv); use another name for test runs
#                       so they do not overwrite real results
#
# Output, in data/ghsl/:
#   <slug>_ghs_built_s_2018_10m.tif   built-up surface (m2 per 10 m cell, 0-100),
#                                     cut to the city boundary, in the city's
#                                     UTM zone (native Mollweide cannot be
#                                     exported by Earth Engine)
#   ghsl_capitals_summary.csv         one row per city, including sanity checks
#
# Data: GHSL, European Commission JRC, licensed CC BY 4.0 (credit the source).

# ---- Configuration ----------------------------------------------------------

ghsl_year <- 2018
ghsl_asset <- paste0("JRC/GHSL/P2023A/GHS_BUILT_S_10m/", ghsl_year)
ghsl_band <- "built_surface"
cell_size_m <- 10

# Capitals of the southern states. Names must match what data_extraction_osm.R
# was run with, because the boundary file is found from the city name.
# Notes: Hyderabad was the shared capital of Andhra Pradesh until 2024, and
# Amaravati (AP's new capital) was barely built in 2018, so expect very little
# built-up surface there.
capitals <- data.frame(
  state = c("Karnataka", "Tamil Nadu", "Telangana", "Kerala", "Andhra Pradesh"),
  city = c(
    "Bengaluru, India", "Chennai, India", "Hyderabad, India",
    "Thiruvananthapuram, India", "Amaravati, India"
  ),
  stringsAsFactors = FALSE
)

# ---- Helpers (no package needed, so they can be tested on their own) --------

# Must match the slug rule in data_extraction_osm.R.
city_slug <- function(city) {
  gsub("[^a-z]", "", tolower(strsplit(city, ",")[[1]][1]))
}

# Grid of lon/lat tiles covering a bounding box. Tile edges sit on multiples of
# `step`, so tiles from different cities line up and can be reused.
tile_grid <- function(xmin, ymin, xmax, ymax, step) {
  ix <- seq(floor(xmin / step), max(floor(xmin / step), ceiling(xmax / step) - 1))
  iy <- seq(floor(ymin / step), max(floor(ymin / step), ceiling(ymax / step) - 1))
  g <- expand.grid(ix = ix, iy = iy)
  g$xmin <- g$ix * step
  g$xmax <- (g$ix + 1) * step
  g$ymin <- g$iy * step
  g$ymax <- (g$iy + 1) * step
  g
}

summary_row <- function(city, state, status, tiles = NA_integer_,
                        boundary_area_km2 = NA_real_, raster_area_km2 = NA_real_,
                        built_up_km2 = NA_real_, built_up_share = NA_real_,
                        file = NA_character_, message = "") {
  data.frame(
    city = city, state = state, status = status, tiles = tiles,
    boundary_area_km2 = boundary_area_km2, raster_area_km2 = raster_area_km2,
    built_up_km2 = built_up_km2, built_up_share = built_up_share,
    file = file, message = message, stringsAsFactors = FALSE
  )
}

# ---- Earth Engine -----------------------------------------------------------

init_ee <- function(project) {
  if (!nzchar(project) || grepl("YOUR", toupper(project), fixed = TRUE)) {
    stop("Pass your real Google Cloud project ID: --project=<id> (or set EE_PROJECT).",
         call. = FALSE)
  }
  # Newer reticulate installs Python packages on demand; this has to be declared
  # before Python starts. Older reticulate lacks it and needs py_install instead.
  if ("py_require" %in% getNamespaceExports("reticulate")) {
    tryCatch(reticulate::py_require(c("earthengine-api", "requests")), error = function(e) NULL)
  }
  ee <- tryCatch(
    reticulate::import("ee"),
    error = function(e) {
      stop("Python package 'earthengine-api' not found (", conditionMessage(e), "). ",
           "Update reticulate so it can install it automatically, or install ",
           "Python 3.12 and run reticulate::py_install(\"earthengine-api\").",
           call. = FALSE)
    }
  )
  ok <- tryCatch({ ee$Initialize(project = project); TRUE }, error = function(e) FALSE)
  if (!ok) {
    message("Earth Engine login needed; follow the prompts.")
    ee$Authenticate()
    ee$Initialize(project = project)
  }
  ee
}

# Earth Engine's getDownloadURL cannot write GeoTIFFs in a projection that only
# exists as WKT text, which is how GHSL's native World_Mollweide is reported
# (HTTP 400 "Unable to write GeoTIFFs in projection"). In that case tiles are
# downloaded in the UTM zone of the city instead: a standard EPSG code that
# Earth Engine accepts, metric, and equal-area to within ~0.1% across a city.
utm_epsg <- function(lon, lat) {
  zone <- floor((lon + 180) / 6) + 1
  paste0("EPSG:", (if (lat >= 0) 32600 else 32700) + zone)
}

# Pixel-exact request for a lon/lat tile: snap it to the layer's own 10 m grid and
# describe it by affine transform + size. No resampling, and tiles always line up.
# `transform` is the projection transform reported by Earth Engine
# [xScale, xShear, xTranslation, yShear, yScale, yTranslation]. Returns NULL if
# it does not look like a 10 m north-up grid.
native_grid <- function(xmin, ymin, xmax, ymax, crs, transform, cell = cell_size_m) {
  transform <- suppressWarnings(as.numeric(unlist(transform)))
  if (length(transform) != 6 || anyNA(transform) ||
      abs(transform[1] - cell) > 1e-6 || abs(transform[5] + cell) > 1e-6) {
    return(NULL)
  }
  # Walk along the tile edges so the curved edges in Mollweide are covered.
  xs <- seq(xmin, xmax, length.out = 21)
  ys <- seq(ymin, ymax, length.out = 21)
  pts <- data.frame(
    x = c(xs, xs, rep(xmin, 21), rep(xmax, 21)),
    y = c(rep(ymin, 21), rep(ymax, 21), ys, ys)
  )
  m <- sf::st_coordinates(sf::st_transform(
    sf::st_as_sf(pts, coords = c("x", "y"), crs = 4326), sf::st_crs(crs)))
  ox <- transform[3]
  oy <- transform[6]
  x0 <- ox + floor((min(m[, 1]) - ox) / cell) * cell
  y1 <- oy - floor((oy - max(m[, 2])) / cell) * cell
  w <- ceiling((max(m[, 1]) - x0) / cell)
  h <- ceiling((y1 - min(m[, 2])) / cell)
  if (w < 1 || h < 1 || w > 10000 || h > 10000) return(NULL)
  list(crs_transform = c(cell, 0, 0, -cell, x0, y1),
       dimensions = c(as.integer(w), as.integer(h)))
}

# Fetch a URL with Python's requests, saving the body to `path` when it succeeds.
# Done in Python so a failure returns Google's own error text, which R's
# download.file() throws away. Returns list(status, text).
fetch_url <- function(url, path) {
  reticulate::py_run_string("
def ghsl_fetch(url, path):
    import requests
    r = requests.get(url, timeout=900)
    if r.status_code == 200:
        with open(path, 'wb') as f:
            f.write(r.content)
        return [r.status_code, '']
    return [r.status_code, r.text[:600]]
")
  res <- reticulate::py$ghsl_fetch(url, path)
  list(status = as.integer(res[[1]]), text = trimws(gsub("\\s+", " ", res[[2]])))
}

# Download one lon/lat tile as a GeoTIFF in the image's native projection.
# Tries a pixel-exact request first, then a region + scale request.
download_tile <- function(ee, img, crs, transform, xmin, ymin, xmax, ymax, dest) {
  region <- ee$Geometry$Rectangle(c(xmin, ymin, xmax, ymax),
                                  proj = "EPSG:4326", geodesic = FALSE)
  variants <- list()
  ng <- tryCatch(native_grid(xmin, ymin, xmax, ymax, crs, transform),
                 error = function(e) NULL)
  if (!is.null(ng)) {
    variants$native_grid <- list(crs = crs, crs_transform = ng$crs_transform,
                                 dimensions = ng$dimensions, format = "GEO_TIFF")
  }
  variants$region_scale <- list(region = region, scale = cell_size_m, crs = crs,
                                format = "GEO_TIFF")

  errors <- character()
  for (v in names(variants)) {
    for (attempt in 1:3) {
      out <- tryCatch({
        url <- img$getDownloadURL(variants[[v]])
        tmp <- tempfile(fileext = ".bin")
        r <- fetch_url(url, tmp)
        if (r$status != 200L) stop("HTTP ", r$status, ": ", r$text, call. = FALSE)

        # Normally a plain GeoTIFF; unzip in case Earth Engine returns a zip.
        if (identical(readBin(tmp, "raw", 2), as.raw(c(0x50, 0x4b)))) {
          td <- tempfile(); dir.create(td)
          utils::unzip(tmp, exdir = td)
          tif <- list.files(td, pattern = "\\.tif$", full.names = TRUE)[1]
          file.copy(tif, tmp, overwrite = TRUE)
        }
        terra::rast(tmp) # fails here if the download is not a valid raster
        file.copy(tmp, dest, overwrite = TRUE)
        TRUE
      }, error = function(e) conditionMessage(e))

      if (isTRUE(out)) {
        if (v != names(variants)[1]) message("  tile fetched with the '", v, "' request style")
        return(invisible(TRUE))
      }
      errors[v] <- out
      # Retry only temporary problems (rate limit, server error); a 400 will not fix itself.
      if (!grepl("HTTP (429|5[0-9][0-9])", out)) break
      Sys.sleep(2^attempt)
    }
    message("  '", v, "' request failed: ", out)
  }
  stop("Tile download failed.\n", paste0("  ", names(errors), ": ", errors, collapse = "\n"),
       call. = FALSE)
}

# ---- Mosaic, cut to the boundary, summarise (needs sf + terra, no Earth Engine)

mosaic_and_cut <- function(tile_files, boundary) {
  rs <- lapply(tile_files, terra::rast)

  # Tiles request the same native grid, so they should line up exactly.
  ref <- terra::ext(rs[[1]])
  off <- vapply(rs, function(r) {
    e <- terra::ext(r)
    max(abs((e$xmin - ref$xmin) / cell_size_m - round((e$xmin - ref$xmin) / cell_size_m)),
        abs((e$ymax - ref$ymax) / cell_size_m - round((e$ymax - ref$ymax) / cell_size_m)))
  }, numeric(1))
  if (any(off > 1e-3)) {
    warning("Tiles are not on one pixel grid; merging will resample some of them.",
            call. = FALSE)
  }

  r <- if (length(rs) == 1) rs[[1]] else do.call(terra::merge, rs)
  names(r) <- "built_surface_m2"

  v <- terra::project(terra::vect(boundary), terra::crs(r))
  r <- terra::mask(terra::crop(r, v), v)

  cell_area <- prod(terra::res(r))
  n_cells <- terra::global(!is.na(r), "sum")[[1]]
  total_m2 <- terra::global(r, "sum", na.rm = TRUE)[[1]]
  list(
    raster = r,
    raster_area_km2 = n_cells * cell_area / 1e6,
    built_up_km2 = total_m2 / 1e6,
    built_up_share = total_m2 / (n_cells * cell_area)
  )
}

# ---- Per-city ---------------------------------------------------------------

process_city <- function(ee, img, crs, city, state, opts) {
  slug <- city_slug(city)
  message("\n=== ", city, " (", state, ") ===")
  out_tif <- file.path(opts$ghsl_dir, paste0(slug, "_ghs_built_s_", ghsl_year, "_10m.tif"))
  if (!opts$force && file.exists(out_tif)) {
    message("Output exists, skipping (use --force to rebuild).")
    return(summary_row(city, state, "skipped", file = out_tif))
  }

  b_file <- file.path(opts$data_dir, paste0(slug, "_boundary.gpkg"))
  if (!file.exists(b_file)) {
    return(summary_row(city, state, "no_boundary", message = paste0(
      "Missing ", b_file, ". Run data_extraction_osm.R for this city first ",
      "(or use --extract-missing).")))
  }

  boundary <- sf::st_transform(sf::st_read(b_file, quiet = TRUE), 4326)
  area_km2 <- as.numeric(sum(sf::st_area(sf::st_transform(boundary, 6933)))) / 1e6
  message("Boundary area: ", round(area_km2), " km2")
  if (area_km2 > 1500) {
    warning(city, ": boundary is ", round(area_km2), " km2, which is large for a ",
            "city. The OSM search may have returned a district or region instead ",
            "of the city; check data/", slug, "_boundary.gpkg.", call. = FALSE)
  }

  bb <- sf::st_bbox(boundary)
  grid <- tile_grid(bb[["xmin"]], bb[["ymin"]], bb[["xmax"]], bb[["ymax"]], opts$tile_deg)
  tile_polys <- sf::st_sfc(lapply(seq_len(nrow(grid)), function(i) {
    sf::st_polygon(list(rbind(
      c(grid$xmin[i], grid$ymin[i]), c(grid$xmax[i], grid$ymin[i]),
      c(grid$xmax[i], grid$ymax[i]), c(grid$xmin[i], grid$ymax[i]),
      c(grid$xmin[i], grid$ymin[i])
    )))
  }), crs = 4326)
  grid <- grid[lengths(sf::st_intersects(tile_polys, boundary)) > 0, ]
  message(nrow(grid), " tile(s) to fetch at ", opts$tile_deg, " degrees")

  # Download CRS: the native one if Earth Engine can write it (an EPSG code),
  # otherwise the city's UTM zone on a 10 m grid anchored at the origin, so all
  # tiles of the city still sit on one pixel grid.
  dl_crs <- crs
  dl_transform <- opts$native_transform
  if (!grepl("^EPSG:[0-9]+$", crs)) {
    cen <- sf::st_coordinates(sf::st_centroid(sf::st_union(boundary)))
    dl_crs <- utm_epsg(cen[1, 1], cen[1, 2])
    dl_transform <- c(cell_size_m, 0, 0, 0, -cell_size_m, 0)
    message("Native CRS cannot be exported by Earth Engine; downloading in ", dl_crs)
  }

  tile_dir <- file.path(opts$ghsl_dir, "tiles", slug)
  dir.create(tile_dir, recursive = TRUE, showWarnings = FALSE)
  tile_files <- file.path(tile_dir, sprintf("tile_%d_%d.tif", grid$ix, grid$iy))
  for (i in seq_len(nrow(grid))) {
    if (file.exists(tile_files[i])) next
    message("  downloading tile ", i, "/", nrow(grid))
    download_tile(ee, img, dl_crs, dl_transform,
                  grid$xmin[i], grid$ymin[i], grid$xmax[i], grid$ymax[i],
                  tile_files[i])
  }

  res <- mosaic_and_cut(tile_files, boundary)
  terra::writeRaster(res$raster, out_tif, overwrite = TRUE,
                     gdal = c("COMPRESS=DEFLATE"))

  # The grid is (near) equal-area, so the number of boundary cells times the
  # cell area should match the boundary area. A big gap means a mask/CRS problem.
  gap <- abs(res$raster_area_km2 - area_km2) / area_km2
  if (gap > 0.1) {
    warning(city, ": raster area (", round(res$raster_area_km2),
            " km2) differs from boundary area (", round(area_km2), " km2) by ",
            round(100 * gap), "%.", call. = FALSE)
  }
  message("Built-up: ", round(res$built_up_km2, 1), " km2 (",
          round(100 * res$built_up_share, 1), "% of the city)")

  summary_row(city, state, "ok", tiles = nrow(grid),
              boundary_area_km2 = round(area_km2, 1),
              raster_area_km2 = round(res$raster_area_km2, 1),
              built_up_km2 = round(res$built_up_km2, 2),
              built_up_share = round(res$built_up_share, 4), file = out_tif)
}

# ---- Main -------------------------------------------------------------------

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  opt <- function(name, default) {
    hit <- grep(paste0("^--", name, "="), args, value = TRUE)
    if (length(hit)) sub(paste0("^--", name, "="), "", hit[1]) else default
  }

  for (p in c("reticulate", "sf", "terra")) {
    if (!requireNamespace(p, quietly = TRUE)) {
      stop("Please install the '", p, "' package.", call. = FALSE)
    }
  }
  old_s2 <- sf::sf_use_s2(FALSE)
  on.exit(sf::sf_use_s2(old_s2), add = TRUE)

  root <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()
  opts <- list(
    data_dir = file.path(root, "data"),
    ghsl_dir = file.path(root, "data", "ghsl"),
    force = "--force" %in% args,
    tile_deg = as.numeric(opt("tile-deg", "0.1"))
  )
  dir.create(opts$ghsl_dir, recursive = TRUE, showWarnings = FALSE)

  cities_arg <- args[!startsWith(args, "--")]
  todo <- if (length(cities_arg)) {
    data.frame(state = NA_character_, city = cities_arg, stringsAsFactors = FALSE)
  } else {
    capitals
  }

  # Optionally run the OSM extraction for cities that have no boundary yet.
  if ("--extract-missing" %in% args) {
    missing <- todo$city[!file.exists(file.path(
      opts$data_dir, paste0(vapply(todo$city, city_slug, ""), "_boundary.gpkg")))]
    if (length(missing)) {
      message("Running OSM extraction for: ", paste(missing, collapse = "; "))
      old_wd <- setwd(root)
      on.exit(setwd(old_wd), add = TRUE)
      rscript <- file.path(R.home("bin"), "Rscript")
      if (!file.exists(rscript) && !file.exists(paste0(rscript, ".exe"))) {
        rscript <- if (nzchar(Sys.which("Rscript"))) Sys.which("Rscript") else "Rscript"
      }
      system2(rscript,
              c(shQuote(file.path("data_extraction", "data_extraction_osm.R")),
                shQuote(missing)))
    }
  }

  ee <- init_ee(opt("project", Sys.getenv("EE_PROJECT", unset = "")))
  img <- ee$Image(ghsl_asset)$select(ghsl_band)
  # Earth Engine reports a standard projection under `crs` (e.g. "EPSG:4326") but
  # a custom one such as GHSL's Mollweide as WKT text under `wkt`. Either can be
  # passed back to getDownloadURL as `crs`.
  pinfo <- img$projection()$getInfo()
  crs <- if (!is.null(pinfo$crs)) pinfo$crs else pinfo$wkt
  if (is.null(crs) || !nzchar(crs)) {
    stop("Could not read the projection of ", ghsl_asset, ". Earth Engine returned ",
         "these fields: ", paste(names(pinfo), collapse = ", "), call. = FALSE)
  }
  opts$native_transform <- unlist(pinfo$transform)
  message("Asset ", ghsl_asset, ", band ", ghsl_band, ", native CRS: ",
          gsub("\\s+", " ", substr(crs, 1, 70)))

  rows <- lapply(seq_len(nrow(todo)), function(i) {
    tryCatch(
      process_city(ee, img, crs, todo$city[i], todo$state[i], opts),
      error = function(e) {
        message("FAILED: ", todo$city[i], ": ", conditionMessage(e))
        summary_row(todo$city[i], todo$state[i], "error", message = conditionMessage(e))
      }
    )
  })
  summary_tbl <- do.call(rbind, rows)
  print(summary_tbl[, setdiff(names(summary_tbl), c("file", "message"))])
  utils::write.csv(summary_tbl,
                   file.path(opts$ghsl_dir, opt("summary", "ghsl_capitals_summary.csv")),
                   row.names = FALSE)
  invisible(summary_tbl)
}

# Run only when executed as a script, not when sourced (e.g. for testing).
if (sys.nframe() == 0L) {
  res <- main()
  # Non-zero exit status so run_pipeline.R reports the step as FAILED.
  if (any(res$status %in% c("error", "no_boundary"))) quit(status = 1L)
}
