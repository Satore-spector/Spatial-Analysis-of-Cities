library(sf)
library(dplyr)
library(ggplot2)

# Usage (from the project root): Rscript data_extraction/amenity_wrap.R [slug]
# slug is the lower-case city name used by data_extraction_osm.R (default: bengaluru)
args <- commandArgs(trailingOnly = TRUE)
slug <- if (length(args) >= 1) args[1] else "bengaluru"
city_title <- tools::toTitleCase(slug)

root <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()
data_dir <- file.path(root, "data")
fig_dir <- file.path(data_dir, "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

b <- st_read(file.path(data_dir, paste0(slug, "_boundary.gpkg")), quiet = TRUE)
p <- st_read(file.path(data_dir, paste0(slug, "_pois.gpkg")), quiet = TRUE)

# Group raw OSM tags into broader categories (order matters: first match wins)
p <- p |>
    mutate(category = case_when(
        amenity %in% c(
            "restaurant", "cafe", "fast_food", "bar", "pub", "food_court",
            "ice_cream", "biergarten"
        ) ~ "Food & drink",
        !is.na(shop) ~ "Retail",
        amenity %in% c("bank", "atm", "bureau_de_change", "money_transfer") ~ "Finance",
        !is.na(office) ~ "Offices",
        amenity %in% c(
            "hospital", "clinic", "doctors", "dentist", "pharmacy",
            "veterinary"
        ) | !is.na(healthcare) ~ "Health",
        amenity %in% c(
            "school", "college", "university", "kindergarten",
            "library", "coaching", "training"
        ) ~ "Education",
        amenity %in% c("place_of_worship") ~ "Worship",
        !is.na(public_transport) | !is.na(railway) |
            amenity %in% c(
                "bus_station", "taxi", "fuel", "parking",
                "bicycle_rental", "charging_station"
            ) ~ "Transport",
        !is.na(leisure) | !is.na(tourism) |
            amenity %in% c(
                "cinema", "theatre", "community_centre",
                "arts_centre", "marketplace"
            ) ~ "Leisure & culture",
        TRUE ~ "Other"
    ))

# How many of each? ("Other" should not dominate; if it does, inspect it)
print(sort(table(p$category), decreasing = TRUE))
print(head(sort(table(p$amenity[p$category == "Other"]), decreasing = TRUE), 20))

pal <- c(
    "Food & drink" = "#E4572E", "Retail" = "#2E86AB", "Finance" = "#F6AE2D",
    "Offices" = "#6A4C93", "Health" = "#2BA84A", "Education" = "#17BEBB",
    "Worship" = "#8D6E63", "Transport" = "#222222",
    "Leisure & culture" = "#F28CB1", "Other" = "#BBBBBB"
)

# 1. All categories on one map
ggplot() +
    geom_sf(data = b, fill = NA, colour = "grey30") +
    geom_sf(data = p, aes(colour = category), size = 0.15, alpha = 0.6) +
    scale_colour_manual(values = pal) +
    guides(colour = guide_legend(override.aes = list(size = 3, alpha = 1))) +
    theme_void() +
    labs(title = paste(city_title, "POIs by category"), colour = NULL)
ggsave(file.path(fig_dir, paste0(slug, "_pois_by_category.png")),
    width = 9, height = 9, dpi = 200
)

# 2. One panel per category (easier to read than one crowded map)
ggplot() +
    geom_sf(data = b, fill = NA, colour = "grey60", linewidth = 0.2) +
    geom_sf(
        data = p, aes(colour = category), size = 0.1, alpha = 0.5,
        show.legend = FALSE
    ) +
    scale_colour_manual(values = pal) +
    facet_wrap(~category, ncol = 3) +
    theme_void()
ggsave(file.path(fig_dir, paste0(slug, "_pois_facets.png")),
    width = 12, height = 12, dpi = 200
)
