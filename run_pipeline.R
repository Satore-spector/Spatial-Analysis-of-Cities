# Run the whole data pipeline in order, or test it one stage at a time.
#
# Pipeline order:
#   1. OSM extraction      data_extraction/data_extraction_osm.R
#   2. POI category plots  data_extraction/amenitiy_wrap.r        (per city)
#   3. GHSL built-up       data_extraction/ghsl_builtup_capitals.R
#
# HOW TO TEST (cheapest first; run from the project root):
#   Rscript run_pipeline.R --check
#       Checks that packages, Python and settings are in place. No network.
#   Rscript run_pipeline.R --test
#       Offline test of the GHSL cutting/maths on fake data. No login needed.
#   Rscript run_pipeline.R --smoke-ghsl --project=MY_GCP_PROJECT
#       One tiny real Earth Engine download (a 1.5 km circle in central Bengaluru)
#       to prove login, project and download work. About a minute.
#   Rscript run_pipeline.R --project=MY_GCP_PROJECT "Bengaluru, India"
#       The full pipeline for one real city.
#   Rscript run_pipeline.R --project=MY_GCP_PROJECT
#       The full pipeline for the five southern capitals.
#
# Options:
#   --project=ID    Google Cloud project for Earth Engine (or set env var EE_PROJECT)
#   --skip-osm      skip step 1 (use files already in data/)
#   --skip-plots    skip step 2
#   --skip-ghsl     skip step 3
#   --force         redo outputs that already exist (steps 1 and 3)
#   --strict        step 1: stop if a boundary falls back to a bounding box
#   --tile-deg=0.1  step 3: size of download tiles in degrees
# City names ("Pune, India") can be listed after the options. With none, the
# southern capitals from ghsl_builtup_capitals.R are used.
#
# Cities that fail in step 1 are dropped from the later steps, and the run
# ends with a table of what happened. Exit status is 1 if any step failed.

# Path to Rscript: next to R itself if it is there, otherwise whatever is on the PATH.
find_rscript <- function() {
  p <- file.path(R.home("bin"), "Rscript")
  if (file.exists(p) || file.exists(paste0(p, ".exe"))) return(p)
  w <- Sys.which("Rscript")
  if (nzchar(w)) w else "Rscript"
}
rscript <- find_rscript()
quote_all <- function(x) vapply(x, shQuote, character(1), USE.NAMES = FALSE)

# Run one R script as a separate process; returns its exit status.
run_step <- function(label, script, args = character()) {
  message("\n", strrep("#", 70), "\n# ", label, "\n", strrep("#", 70))
  status <- suppressWarnings(
    system2(rscript, c(shQuote(script), quote_all(args)))
  )
  as.integer(status)
}

# What is installed? Returns a data frame: check, ok, fix.
preflight <- function(root, need_osm, need_plots, need_ghsl, project) {
  rows <- list()
  add <- function(check, ok, fix = "") {
    rows[[length(rows) + 1]] <<- data.frame(check = check, ok = isTRUE(ok), fix = fix,
                                            stringsAsFactors = FALSE)
  }
  has_pkg <- function(p) requireNamespace(p, quietly = TRUE)

  scripts <- c(
    if (need_osm) "data_extraction/data_extraction_osm.R",
    if (need_plots) "data_extraction/amenitiy_wrap.r",
    if (need_ghsl) "data_extraction/ghsl_builtup_capitals.R"
  )
  for (s in scripts) add(paste("file", s), file.exists(file.path(root, s)),
                         "run from the project root, and apply the patches")

  pkgs <- unique(c(
    if (need_osm) c("osmextract", "osmdata", "sf", "dplyr"),
    if (need_plots) c("sf", "dplyr", "ggplot2"),
    if (need_ghsl) c("sf", "terra", "reticulate")
  ))
  for (p in pkgs) add(paste("R package", p), has_pkg(p),
                      paste0("install.packages(\"", p, "\")"))

  if (need_ghsl) {
    add("Earth Engine project set",
        nzchar(project) && !grepl("YOUR", toupper(project), fixed = TRUE),
        "pass your real Google Cloud project ID: --project=<id> (or set EE_PROJECT)")
    if (has_pkg("reticulate")) {
      # Newer reticulate installs Python packages on demand. This must run
      # before Python starts, so it comes before the check below.
      if ("py_require" %in% getNamespaceExports("reticulate")) {
        tryCatch(reticulate::py_require("earthengine-api"), error = function(e) NULL)
      }
      add("Python package earthengine-api",
          tryCatch(reticulate::py_module_available("ee"), error = function(e) FALSE),
          paste0("update reticulate (install.packages(\"reticulate\")) so it can install ",
                 "Python packages itself, or install Python 3.12 and run ",
                 "reticulate::py_install(\"earthengine-api\")"))
    }
  }
  do.call(rbind, rows)
}

print_preflight <- function(pf) {
  for (i in seq_len(nrow(pf))) {
    message(if (pf$ok[i]) "  [ok]      " else "  [MISSING] ", pf$check[i],
            if (!pf$ok[i] && nzchar(pf$fix[i])) paste0("   -> ", pf$fix[i]) else "")
  }
  invisible(all(pf$ok))
}

# Tiny real Earth Engine job: a 1.5 km circle in central Bengaluru. Needs no OSM.
smoke_ghsl <- function(root, project, ghsl_script) {
  data_dir <- file.path(root, "data")
  dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
  centre <- sf::st_sfc(sf::st_point(c(77.5946, 12.9716)), crs = 4326)
  circle <- sf::st_transform(sf::st_buffer(sf::st_transform(centre, 6933), 1500), 4326)
  sf::st_write(sf::st_sf(geometry = sf::st_cast(circle, "MULTIPOLYGON")),
               file.path(data_dir, "smoketest_boundary.gpkg"),
               delete_dsn = TRUE, quiet = TRUE)

  status <- run_step("Smoke test: GHSL download for a 1.5 km circle", ghsl_script,
                     c("Smoketest, India", paste0("--project=", project), "--force",
                       "--summary=ghsl_smoketest_summary.csv"))
  csv <- file.path(data_dir, "ghsl", "ghsl_smoketest_summary.csv")
  res <- if (file.exists(csv)) utils::read.csv(csv) else NULL
  pass <- identical(status, 0L) && !is.null(res) && nrow(res) == 1 &&
    identical(res$status[1], "ok") && isTRUE(res$built_up_km2[1] > 0.5)
  if (pass) {
    message("\nSMOKE TEST PASSED: ", res$built_up_km2[1], " km2 built-up in ",
            res$raster_area_km2[1], " km2 (expected area is about 7 km2).")
  } else {
    message("\nSMOKE TEST FAILED. See the messages above",
            if (!is.null(res)) paste0(" (summary says: ", res$status[1], " ", res$message[1], ")"),
            ".")
  }
  pass
}

main <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  flag <- function(name) paste0("--", name) %in% args
  opt <- function(name, default = "") {
    hit <- grep(paste0("^--", name, "="), args, value = TRUE)
    if (length(hit)) sub(paste0("^--", name, "="), "", hit[1]) else default
  }

  root <- if (requireNamespace("here", quietly = TRUE)) here::here() else getwd()
  if (!file.exists(file.path(root, "data_extraction", "data_extraction_osm.R"))) {
    stop("Run this from the project root (the folder that contains data_extraction/).",
         call. = FALSE)
  }
  setwd(root)

  osm_script <- file.path("data_extraction", "data_extraction_osm.R")
  plot_script <- file.path("data_extraction", "amenitiy_wrap.r")
  ghsl_script <- file.path("data_extraction", "ghsl_builtup_capitals.R")
  project <- opt("project", Sys.getenv("EE_PROJECT", unset = ""))

  # ---- Test modes ----
  if (flag("check")) {
    message("Checking setup in ", root)
    pf <- preflight(root, TRUE, TRUE, TRUE, project)
    ok <- print_preflight(pf)
    message(if (ok) "\nEverything needed is in place." else "\nFix the [MISSING] items above.")
    quit(status = if (ok) 0L else 1L)
  }
  if (flag("test")) {
    status <- run_step("Offline test of the GHSL processing (fake data)",
                       file.path("tests", "test_ghsl_offline.R"))
    quit(status = if (identical(status, 0L)) 0L else 1L)
  }
  if (flag("smoke-ghsl")) {
    pf <- preflight(root, FALSE, FALSE, TRUE, project)
    if (!print_preflight(pf)) quit(status = 1L)
    quit(status = if (smoke_ghsl(root, project, ghsl_script)) 0L else 1L)
  }

  # ---- Full pipeline ----
  skip_osm <- flag("skip-osm"); skip_plots <- flag("skip-plots"); skip_ghsl <- flag("skip-ghsl")
  pf <- preflight(root, !skip_osm, !skip_plots, !skip_ghsl, project)
  if (!print_preflight(pf)) {
    message("\nNothing was run. Fix the [MISSING] items above (or use --skip-... flags).")
    quit(status = 1L)
  }

  # Reuse the city list and slug rule from the GHSL script (it does not run when sourced).
  env <- new.env()
  sys.source(ghsl_script, envir = env)
  cities <- args[!startsWith(args, "--")]
  if (length(cities) == 0) cities <- env$capitals$city
  slug <- vapply(cities, env$city_slug, character(1), USE.NAMES = FALSE)
  data_dir <- "data"
  message("\nCities: ", paste(cities, collapse = "; "))

  steps_log <- data.frame(step = character(), status = character(), minutes = numeric(),
                    stringsAsFactors = FALSE)
  record <- function(step, status, t0) {
    steps_log[nrow(steps_log) + 1, ] <<- list(step, status, round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1))
  }
  finish <- function() {
    message("\n", strrep("=", 70), "\nPIPELINE SUMMARY")
    print(steps_log, row.names = FALSE)
    quit(status = if (any(steps_log$status == "FAILED")) 1L else 0L)
  }

  # Step 1: OSM extraction
  if (skip_osm) {
    record("1 OSM extraction", "skipped", Sys.time())
  } else {
    t0 <- Sys.time()
    st <- run_step("Step 1: OSM extraction", osm_script,
                   c(cities, if (flag("force")) "--force", if (flag("strict")) "--strict"))
    record("1 OSM extraction", if (identical(st, 0L)) "ok" else "FAILED", t0)
    if (!identical(st, 0L)) { message("Step 1 crashed; stopping."); finish() }

    # The script handles per-city errors itself, so read its summary.
    sfile <- file.path(data_dir, "extraction_summary.csv")
    if (file.exists(sfile)) {
      s <- utils::read.csv(sfile, stringsAsFactors = FALSE)
      bad <- s[s$status == "error", ]
      for (i in seq_len(nrow(bad))) message("  extraction failed for ", bad$city[i], ": ", bad$message[i])
      if (nrow(bad) > 0) {
        steps_log$status[nrow(steps_log)] <- paste0("ok, ", nrow(bad), " city failed")
      }
    }
  }

  have_boundary <- file.exists(file.path(data_dir, paste0(slug, "_boundary.gpkg")))
  have_pois <- file.exists(file.path(data_dir, paste0(slug, "_pois.gpkg")))
  if (any(!have_boundary)) {
    message("Dropping from later steps (no boundary file): ",
            paste(cities[!have_boundary], collapse = "; "))
  }

  # Step 2: POI plots (nice to have: a failure here does not stop step 3)
  if (skip_plots) {
    record("2 POI plots", "skipped", Sys.time())
  } else {
    t0 <- Sys.time()
    sts <- vapply(slug[have_pois], function(s) run_step(paste("Step 2: POI plots for", s), plot_script, s),
                  integer(1))
    record("2 POI plots", if (length(sts) && any(sts != 0L)) "FAILED" else "ok", t0)
  }

  # Step 3: GHSL built-up surface
  if (skip_ghsl) {
    record("3 GHSL built-up", "skipped", Sys.time())
  } else if (!any(have_boundary)) {
    message("No city has a boundary file, so step 3 cannot run.")
    record("3 GHSL built-up", "FAILED", Sys.time())
  } else {
    t0 <- Sys.time()
    st <- run_step("Step 3: GHSL built-up surface", ghsl_script,
                   c(cities[have_boundary], paste0("--project=", project),
                     if (flag("force")) "--force",
                     if (nzchar(opt("tile-deg"))) paste0("--tile-deg=", opt("tile-deg"))))
    record("3 GHSL built-up", if (identical(st, 0L)) "ok" else "FAILED", t0)
    gfile <- file.path(data_dir, "ghsl", "ghsl_capitals_summary.csv")
    if (file.exists(gfile)) {
      message("\nGHSL results (", gfile, "):")
      g <- utils::read.csv(gfile, stringsAsFactors = FALSE)
      print(g[, intersect(c("city", "status", "boundary_area_km2", "built_up_km2", "built_up_share"),
                          names(g))], row.names = FALSE)
    }
  }
  finish()
}

if (sys.nframe() == 0L) main()
