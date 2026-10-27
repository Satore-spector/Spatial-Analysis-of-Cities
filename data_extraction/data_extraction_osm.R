# Extract boundary, POIs, buildings and roads from OSM for one or more cities.
#
# Usage (run from the project root):
#   Rscript data_extraction/data_extraction_osm.R "Bengaluru, India" "Pune, India"
#   Rscript data_extraction/data_extraction_osm.R --force "Delhi, India"
#
# With no city arguments it defaults to Bengaluru.
# Flags:
#   --force   re-extract cities whose output files already exist
#   --strict  stop (instead of warn) if a boundary has to fall back to its bbox
#
# Consistent footprints: to use the same boundary definition for every city
# (e.g. a GHSL Urban Centre or a Functional Urban Area), put a polygon file at
#   data/boundaries/<slug>.gpkg      (slug = lower-case city name, letters only)
# and it is used instead of the Nominatim administrative boundary.

library(osmextract)
library(osmdata)
library(sf)
library(dplyr)

# Regional OSM extracts are hundreds of MB; R's default 60 s download limit
# aborts them part-way. Allow up to an hour.
options(timeout = max(3600, getOption("timeout")))

# ---- Configuration ----------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
force_rerun <- "--force" %in% args
strict <- "--strict" %in% args
cities <- args[!startsWith(args, "--")]
if (length(cities) == 0) {
  cities <- "Bengaluru, India"
  # "Delhi, India", "Mumbai, India", "Hyderabad, India",
  # "Kolkata, India", "Chennai, India", "Pune, India", "Ahmedabad, India"
}

root <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()
out_dir <- file.path(root, "data")
boundary_dir <- file.path(out_dir, "boundaries")
# Downloaded Geofabrik extracts are kept here so each regional file is
# fetched once and reused by every city inside it (the default is a tempdir).
pbf_dir <- file.path(out_dir, "osm_pbf")
for (d in c(out_dir, boundary_dir, pbf_dir)) {
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
}

# Major feature extraction
poi_tags <- c("amenity", "shop", "office", "public_transport",
              "railway", "tourism", "leisure", "healthcare")

# ---- Helpers ----------------------------------------------------------------

city_slug <- function(city) {
  gsub("[^a-z]", "", tolower(strsplit(city, ",")[[1]][1]))
}

# Clean any polygon layer into a single valid MULTIPOLYGON in EPSG:4326.
clean_polygon <- function(x) {
  x |>
    st_geometry() |>
    st_transform(4326) |>
    st_make_valid() |>
    st_collection_extract("POLYGON") |>
    st_union()
}

# Returns list(boundary = sf, source = "file"/"nominatim", fallback = logical).
# Assumes sf_use_s2(FALSE) has been set by the caller.
get_boundary <- function(city, slug) {
  user_file <- file.path(boundary_dir, paste0(slug, ".gpkg"))

  if (file.exists(user_file)) {
    message("Using user-supplied boundary: ", user_file)
    geom <- clean_polygon(st_read(user_file, quiet = TRUE))
    return(list(
      boundary = st_sf(geometry = st_cast(geom, "MULTIPOLYGON")),
      source = "file", fallback = FALSE
    ))
  }

  boundary <- getbb(city, format_out = "sf_polygon", limit = 1)
  geom <- clean_polygon(boundary) |>
    st_simplify(preserveTopology = TRUE, dTolerance = 0.0005) # ~50 m

  # Snap to a 4-decimal grid (~11 m) by rounding coordinates via WKT
  geom <- st_as_sfc(st_as_text(geom, digits = 4), crs = 4326) |>
    st_make_valid() |>
    st_buffer(0) |>
    st_collection_extract("POLYGON") |>
    st_union()

  # Test: does it stay valid after being rounded to 5 decimals (what GDAL sees)?
  rounded <- st_as_sfc(st_as_text(geom, digits = 5), crs = 4326)
  ok <- isTRUE(all(st_is_valid(rounded)))
  message("Boundary valid after rounding: ", ok)

  if (!ok) {
    msg <- paste0(
      "Boundary for '", city, "' is invalid after rounding; ",
      "falling back to its bounding box. The study area is no longer ",
      "comparable with other cities."
    )
    if (strict) stop(msg, call. = FALSE)
    warning(msg, call. = FALSE)
    geom <- st_as_sfc(st_bbox(geom))
    st_crs(geom) <- 4326
  }

  list(
    boundary = st_sf(geometry = st_cast(geom, "MULTIPOLYGON")),
    source = "nominatim", fallback = !ok
  )
}

# Build one row of the summary table (used for ok / skipped / error cities).
summary_row <- function(city, status, pois = NA_integer_, pois_from_areas = NA_integer_,
                        buildings = NA_integer_, roads = NA_integer_,
                        boundary_source = NA_character_, boundary_fallback = NA,
                        boundary_area_km2 = NA_real_, message = "") {
  tibble(
    city = city, status = status, pois = pois, pois_from_areas = pois_from_areas,
    buildings = buildings, roads = roads, boundary_source = boundary_source,
    boundary_fallback = boundary_fallback, boundary_area_km2 = boundary_area_km2,
    message = message
  )
}

# Standardise POIs (from nodes or from area centroids) to one schema.
as_poi_sf <- function(x, geom, source) {
  df <- st_drop_geometry(x)
  df$osm_id <- as.character(df$osm_id)
  df <- df[, c("osm_id", "name", poi_tags)]
  df$poi_source <- source
  st_sf(df, geometry = geom)
}

# ---- Per-city extraction ----------------------------------------------------

extract_city <- function(city) {
  slug <- city_slug(city)
  message("\n=== ", city, " ===")

  files <- file.path(out_dir, paste0(slug, c("_boundary", "_pois", "_buildings", "_roads"), ".gpkg"))
  if (!force_rerun && all(file.exists(files))) {
    message("Outputs already exist, skipping (use --force to redo).")
    return(summary_row(city, "skipped"))
  }

  old_s2 <- sf_use_s2(FALSE)
  on.exit(sf_use_s2(old_s2), add = TRUE)

  # 1. Boundary
  b <- get_boundary(city, slug)
  boundary <- b$boundary
  place <- st_point_on_surface(st_geometry(boundary))
  # Equal-area projection so areas are comparable across cities
  area_km2 <- as.numeric(sum(st_area(st_transform(boundary, 6933)))) / 1e6

  st_write(boundary, files[1], delete_dsn = TRUE, quiet = TRUE)

  clip <- list(boundary = boundary, boundary_type = "clipsrc")
  # oe_get picks the smallest Geofabrik extract containing `place` and reuses
  # the downloaded .pbf from pbf_dir on later calls.
  dl <- list(download_directory = pbf_dir, max_file_size = 5e9)

  # 2a. POIs mapped as nodes: shops, amenities, offices, transit, leisure, health.
  # NOTE: the WHERE clause must cover every tag the category scheme in
  # amenity_wrap.R uses, otherwise those categories are undercounted.
  node_pois <- do.call(oe_get, c(list(
    place = place, layer = "points", extra_tags = poi_tags,
    query = "SELECT osm_id, name, amenity, shop, office, public_transport,
                    railway, tourism, leisure, healthcare, geometry
             FROM points
             WHERE amenity IS NOT NULL OR shop IS NOT NULL OR office IS NOT NULL
                OR public_transport IS NOT NULL OR railway IS NOT NULL
                OR tourism IS NOT NULL OR leisure IS NOT NULL
                OR healthcare IS NOT NULL",
    quiet = FALSE), dl, clip))

  # 2b + 3. Polygons - can be large, takes a while.
  # One query serves both buildings and "area POIs". Many malls, hospitals,
  # schools and parks are mapped as areas, not nodes, so a nodes-only POI layer
  # misses them. One pass over the layer also avoids reusing the cached
  # multipolygons file with two different tag sets.
  areas <- do.call(oe_get, c(list(
    place = place, layer = "multipolygons", extra_tags = c("building", poi_tags),
    query = "SELECT osm_id, osm_way_id, name, building, amenity, shop, office,
                    public_transport, railway, tourism, leisure, healthcare, geometry
             FROM multipolygons
             WHERE building IS NOT NULL
                OR amenity IS NOT NULL OR shop IS NOT NULL OR office IS NOT NULL
                OR public_transport IS NOT NULL OR railway IS NOT NULL
                OR tourism IS NOT NULL OR leisure IS NOT NULL
                OR healthcare IS NOT NULL",
    quiet = FALSE), dl, clip))

  buildings <- areas[!is.na(areas$building),
                     c("osm_id", "osm_way_id", "name", "building")]

  # Area POIs -> one point each (point on surface stays inside odd shapes).
  is_poi <- Reduce(`|`, lapply(poi_tags, function(t) !is.na(areas[[t]])))
  area_src <- areas[is_poi, ]
  # Ways have osm_way_id, relations have osm_id: keep whichever exists.
  area_src$osm_id <- dplyr::coalesce(as.character(area_src$osm_id),
                                     as.character(area_src$osm_way_id))
  area_pois <- as_poi_sf(area_src, st_point_on_surface(st_geometry(area_src)), "area")

  # A node placed inside a building that is also tagged as the same amenity is
  # counted twice; filter on `poi_source` later if that matters for your analysis.
  pois <- rbind(as_poi_sf(node_pois, st_geometry(node_pois), "node"), area_pois)

  # 4. Road network (lines)
  roads <- do.call(oe_get, c(list(
    place = place, layer = "lines",
    query = "SELECT osm_id, name, highway, geometry
             FROM lines
             WHERE highway IN ('motorway','trunk','primary','secondary',
                               'tertiary','residential','unclassified',
                               'motorway_link','trunk_link','primary_link',
                               'secondary_link','tertiary_link')",
    quiet = FALSE), dl, clip))

  # 5. Save
  st_write(pois,      files[2], delete_dsn = TRUE, quiet = TRUE)
  st_write(buildings, files[3], delete_dsn = TRUE, quiet = TRUE)
  st_write(roads,     files[4], delete_dsn = TRUE, quiet = TRUE)

  summary_row(
    city, "ok",
    pois = nrow(pois), pois_from_areas = nrow(area_pois),
    buildings = nrow(buildings), roads = nrow(roads),
    boundary_source = b$source, boundary_fallback = b$fallback,
    boundary_area_km2 = round(area_km2, 1)
  )
}

# One failing city must not lose the rest of a long multi-city run.
safe_extract <- function(city) {
  tryCatch(
    extract_city(city),
    error = function(e) {
      message("FAILED: ", city, ": ", conditionMessage(e))
      summary_row(city, "error", message = conditionMessage(e))
    }
  )
}

# ---- Run --------------------------------------------------------------------

summary_tbl <- bind_rows(lapply(cities, safe_extract))
print(summary_tbl)
write.csv(summary_tbl, file.path(out_dir, "extraction_summary.csv"), row.names = FALSE)

if (any(summary_tbl$boundary_fallback %in% TRUE)) {
  warning("Some cities used a bbox fallback boundary; see extraction_summary.csv",
          call. = FALSE)
}

# ---- Quick sanity check: plot POIs over the boundary (interactive only) ----
if (interactive()) {
  first_ok <- summary_tbl$city[summary_tbl$status == "ok"][1]
  if (!is.na(first_ok)) {
    slug <- city_slug(first_ok)
    b <- st_read(file.path(out_dir, paste0(slug, "_boundary.gpkg")), quiet = TRUE)
    p <- st_read(file.path(out_dir, paste0(slug, "_pois.gpkg")), quiet = TRUE)
    plot(st_geometry(b), main = paste(first_ok, "POIs"))
    plot(st_geometry(p), add = TRUE, pch = ".", col = "red")
  }
}
