library(osmdata)
library(OpenStreetMap)
library(sf)
library(tidyverse)
library(osmextract)
library(ohsome)
library(mapview
# Get Bangalore (Bengaluru) admin boundary from OSM
blr <- oe_get_boundary("Karnataka", "Bengaluru", quiet = TRUE)

blr |>
    mutate(id = NAME) |> # check names(blr) for the right column
    ohsome_elements_length(
        filter = "highway=* and geometry:line",
        return_value = "density"
    ) |>
    set_time("2025-01-01") |>
    set_grouping("boundary") |>
    ohsome_post() |>
    mapview(zcol = "value", layer.name = "Road length (m/km²)")