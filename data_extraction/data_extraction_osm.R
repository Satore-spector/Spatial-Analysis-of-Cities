
library(osmextract)
library(osmdata)
library(sf)
library(dplyr)

# Cities
cities <- c(
  "Bengaluru, India"
  # "Delhi, India", "Mumbai, India", "Hyderabad, India",
  # "Kolkata, India", "Chennai, India", "Pune, India", "Ahmedabad, India"
)

out_dir <- "C:\\Spatial-Analysis-of-Cities\\data"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

#major feature extraction
poi_tags <- c("amenity", "shop", "office", "public_transport",
              "railway", "tourism", "leisure", "healthcare")

extract_city <- function(city) {
  slug <- gsub("[^a-z]", "", tolower(strsplit(city, ",")[[1]][1]))
  message("\n=== ", city, " ===")

# boundry shape file extraction
  boundary <- getbb(city, format_out = "sf_polygon", limit = 1)
  boundary <- st_transform(boundary, 4326)

  old_s2 <- sf_use_s2(FALSE)
  geom <- boundary |>
      st_geometry() |>
      st_make_valid() |>
      st_union() |>
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
      message("Falling back to bounding box for the clip")
      geom <- st_as_sfc(st_bbox(geom))
      st_crs(geom) <- 4326
  }

  boundary <- st_sf(geometry = st_cast(geom, "MULTIPOLYGON"))
  place <- st_point_on_surface(st_geometry(boundary))
  sf_use_s2(old_s2)

  st_write(boundary, file.path(out_dir, paste0(slug, "_boundary.gpkg")),
      delete_dsn = TRUE, quiet = TRUE
  )

  clip <- list(boundary = boundary, boundary_type = "clipsrc")

  # 2. POIs (points): shops, amenities, offices, transit
  pois <- do.call(oe_get, c(list(
    place = place, layer = "points", extra_tags = poi_tags,
    query = "SELECT osm_id, name, amenity, shop, office, public_transport,
                    railway, tourism, leisure, healthcare, geometry
             FROM points
             WHERE amenity IS NOT NULL OR shop IS NOT NULL OR office IS NOT NULL
                OR public_transport IS NOT NULL OR railway IS NOT NULL",
    quiet = FALSE), clip))

  # 3. Buildings (polygons) - can be large, takes a while
  buildings <- do.call(oe_get, c(list(
    place = place, layer = "multipolygons", extra_tags = "building",
    query = "SELECT osm_id, osm_way_id, name, building, geometry
             FROM multipolygons WHERE building IS NOT NULL",
    quiet = FALSE), clip))

  # 4. Road network (lines)
  roads <- do.call(oe_get, c(list(
    place = place, layer = "lines",
    query = "SELECT osm_id, name, highway, geometry
             FROM lines
             WHERE highway IN ('motorway','trunk','primary','secondary',
                               'tertiary','residential','unclassified',
                               'motorway_link','trunk_link','primary_link',
                               'secondary_link','tertiary_link')",
    quiet = FALSE), clip))

  # 5. Save
  st_write(pois,      file.path(out_dir, paste0(slug, "_pois.gpkg")),      delete_dsn = TRUE, quiet = TRUE)
  st_write(buildings, file.path(out_dir, paste0(slug, "_buildings.gpkg")), delete_dsn = TRUE, quiet = TRUE)
  st_write(roads,     file.path(out_dir, paste0(slug, "_roads.gpkg")),     delete_dsn = TRUE, quiet = TRUE)

  tibble(city = city, pois = nrow(pois), buildings = nrow(buildings), roads = nrow(roads))
}

summary_tbl <- bind_rows(lapply(cities, extract_city))
print(summary_tbl)
write.csv(summary_tbl, file.path(out_dir, "extraction_summary.csv"), row.names = FALSE)

# ---- Quick sanity check: plot POIs over the boundary ----
b <- st_read(file.path(out_dir, "bengaluru_boundary.gpkg"), quiet = TRUE)
p <- st_read(file.path(out_dir, "bengaluru_pois.gpkg"), quiet = TRUE)
plot(st_geometry(b), main = "Bengaluru POIs")
plot(st_geometry(p), add = TRUE, pch = ".", col = "red")