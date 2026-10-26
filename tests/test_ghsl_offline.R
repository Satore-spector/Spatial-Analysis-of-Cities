# Offline test of the GHSL processing in data_extraction/ghsl_builtup_capitals.R.
# Needs only the sf and terra packages: no Earth Engine login, no downloads.
#
# Run from the project root:   Rscript tests/test_ghsl_offline.R
#                         or:   Rscript run_pipeline.R --test
#
# It builds fake 10 m tiles in the same Mollweide projection as the real data
# (every cell = 50 m2 of 100 m2, so the built-up share must come out as 0.5),
# cuts them to a made-up circular city boundary, and checks the area and share.
# It checks the tiling, merging, masking and sums, not the Earth Engine download.

suppressMessages({library(sf); library(terra)})
sf_use_s2(FALSE)

root <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()
script <- file.path(root, "data_extraction", "ghsl_builtup_capitals.R")
if (!file.exists(script)) stop("Run from the project root; cannot find ", script, call. = FALSE)
source(script) # only defines functions; main() does not run when sourced

check <- function(name, ok) {
  cat(sprintf("  %-55s %s\n", name, if (ok) "ok" else "FAILED"))
  if (!ok) stop("Test failed: ", name, call. = FALSE)
}

cat("City names and tile grid\n")
check("slug: Thiruvananthapuram", city_slug("Thiruvananthapuram, India") == "thiruvananthapuram")
check("slug: Bengaluru", city_slug("Bengaluru, India") == "bengaluru")
g <- tile_grid(77.45, 12.83, 77.78, 13.14, 0.1)
check("tiles cover the whole bounding box",
      min(g$xmin) <= 77.45 && max(g$xmax) >= 77.78 && min(g$ymin) <= 12.83 && max(g$ymax) >= 13.14)
check("a box exactly on one tile gives one tile", nrow(tile_grid(77.0, 12.0, 77.1, 12.1, 0.1)) == 1)

cat("Mosaic, mask and sums on fake Mollweide tiles\n")
centre <- st_sfc(st_point(c(77.59, 12.97)), crs = 4326)
boundary <- st_sf(geometry = st_transform(st_buffer(st_transform(centre, 6933), 10000), 4326))
true_km2 <- as.numeric(st_area(st_transform(boundary, 6933))) / 1e6

moll <- "ESRI:54009"
bb <- st_bbox(st_transform(boundary, moll))
x0 <- floor(bb[["xmin"]] / 10) * 10 - 2000
y1 <- ceiling(bb[["ymax"]] / 10) * 10 + 2000
half <- 12000
make_tile <- function(xo, yo) {
  r <- rast(xmin = x0 + xo, xmax = x0 + xo + half, ymax = y1 - yo, ymin = y1 - yo - half,
            resolution = 10, crs = moll)
  values(r) <- 50
  f <- tempfile(fileext = ".tif"); writeRaster(r, f); f
}
files <- c(make_tile(0, 0), make_tile(half, 0), make_tile(0, half), make_tile(half, half))

res <- mosaic_and_cut(files, boundary)
cat(sprintf("  true area %.1f km2, raster area %.1f km2, built-up %.1f km2, share %.3f\n",
            true_km2, res$raster_area_km2, res$built_up_km2, res$built_up_share))
check("raster area within 2% of boundary area",
      abs(res$raster_area_km2 - true_km2) / true_km2 < 0.02)
check("built-up share is 0.5", abs(res$built_up_share - 0.5) < 1e-6)
check("output stays in 10 m Mollweide", all(res(res$raster) == 10) &&
        grepl("Mollweide", crs(res$raster, describe = TRUE)$name))

out <- tempfile(fileext = ".tif")
writeRaster(res$raster, out, overwrite = TRUE, gdal = c("COMPRESS=DEFLATE"))
check("compressed GeoTIFF writes and reads back", all(res(rast(out)) == 10))

one <- mosaic_and_cut(files[1], st_sf(geometry = st_transform(
  st_buffer(st_transform(st_sfc(st_point(c(77.54, 13.02)), crs = 4326), 6933), 3000), 4326)))
check("single-tile case works", one$built_up_km2 > 0)

cat("ALL TESTS PASSED\n")
