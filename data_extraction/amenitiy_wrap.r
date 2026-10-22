library(sf)
library(dplyr)
library(ggplot2)

b <- st_read("C:\\Spatial-Analysis-of-Cities\\data\\bengaluru_boundary.gpkg", quiet = TRUE)
p <- st_read("C:\\Spatial-Analysis-of-Cities\\data\\bengaluru_pois.gpkg", quiet = TRUE)

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
    labs(title = "Bengaluru POIs by category", colour = NULL)
ggsave("data/osm/bengaluru_pois_by_category.png", width = 9, height = 9, dpi = 200)

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
ggsave("C:\\Spatial-Analysis-of-Cities\\data\\gg_wrap.png", width = 12, height = 12, dpi = 200)