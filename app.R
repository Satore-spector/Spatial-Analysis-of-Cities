# ==========================================================
# Spatial Analysis of Cities - results dashboard
# Single continuous-scroll Shiny page, same layout as the
# Financial-Analysis dashboard. Reads the analysis output
# and refreshes itself when you re-run the pipeline.
#
# Put this file in the project root (next to run_country.py).
# It reads  data/india/analysis/classification/
#   city_types.csv, city_types_map.png, report.md
# Override the folder with:  Sys.setenv(SPATIAL_RESULTS_DIR = "path")
#
# Run:  shiny::runApp()
# ==========================================================

library(shiny)
library(ggplot2)
library(scales)

app_dir <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile)), error = function(e) getwd())
results_dir <- Sys.getenv("SPATIAL_RESULTS_DIR",
  unset = file.path(app_dir, "data", "india", "analysis", "classification"))
csv_path <- file.path(results_dir, "city_types.csv")
map_path <- file.path(results_dir, "city_types_map.png")
rep_path <- file.path(results_dir, "report.md")
if (file.exists(map_path)) addResourcePath("res", results_dir)

# ---- Column names: set any of these if auto-detection picks wrong ----
cfg <- list(type = NULL, confidence = NULL, name = NULL, population = NULL)

find_col <- function(df, candidates, override = NULL) {
  if (!is.null(override) && override %in% names(df)) return(override)
  low <- tolower(names(df))
  hit <- candidates[candidates %in% low]
  if (length(hit)) names(df)[match(hit[1], low)] else NA_character_
}

pal <- c("#0F766E", "#0B3B45", "#5EA8A0", "#B45309", "#7C9885", "#334155", "#99B8B4", "#C2410C")

# ---- CSS ----
app_css <- "
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;500;700&display=swap');
:root{--a:#0F766E;--b:#0B3B45;--tint:#EEF6F5;--line:#CFE3E1;--warn:#B45309}
html{font-size:135%;scroll-behavior:smooth}
html,body{margin:0;font-family:'Roboto Mono',monospace;color:#222;background:#fff}
.container-fluid{padding:0!important}
.hero{min-height:100vh;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:40px 8%;background:linear-gradient(160deg,#0F766E,#0B3B45);color:#fff}
.hero-title{font-weight:700;font-size:clamp(1.07rem,3.2vw,2rem);line-height:1.3;max-width:1000px;margin:0}
.hero-sub{margin-top:18px;font-size:.85rem;opacity:.85;max-width:760px;line-height:1.7}
.hero-team{margin-top:36px;font-size:.8rem}
.hero-team b{display:block;letter-spacing:.1em;text-transform:uppercase;margin-bottom:8px}
.dash-section{min-height:100vh;display:flex;flex-direction:column;justify-content:center;gap:12vh;padding:14vh 8%}
.dash-section:nth-of-type(even){background:var(--tint)}
.section-heading{font-weight:700;color:var(--a);border-bottom:3px solid var(--a);display:inline-block;padding-bottom:6px;margin:0 0 26px}
.sub-heading{font-weight:700;color:var(--b);margin:0 0 6px}
.sub-note{margin:0 0 22px;opacity:.75;font-size:.9rem}
.lead{font-size:1.05rem;line-height:1.8;max-width:1200px;margin:0}
.fade-item{will-change:opacity,transform}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:20px}
.stat-card{background:#fff;border:1px solid var(--line);border-radius:22px;box-shadow:0 8px 24px rgba(11,59,69,.08);padding:24px 28px}
.stat-card.warn{border-color:var(--warn)}.stat-card.warn .stat-value{color:var(--warn)}
.stat-value{font-size:1.9rem;font-weight:700;color:var(--a)}
.stat-label{font-weight:500;margin-top:4px}
.stat-note{font-size:.82rem;opacity:.75;margin-top:6px;line-height:1.5}
.callout{background:var(--a);color:#fff;border-radius:22px;padding:26px 32px;margin-top:24px;line-height:1.8;max-width:1400px}
.callout h4{margin:0 0 8px}.callout p{margin:0;font-size:.95rem}
.plot-card{background:#fff;border:1px solid var(--line);border-radius:22px;box-shadow:0 8px 24px rgba(11,59,69,.08);padding:22px 26px}
.plot-card h4{margin:0 0 8px;color:var(--b)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:900px){.two{grid-template-columns:1fr}}
.ev-row{display:grid;grid-template-columns:minmax(160px,.7fr) 1.6fr;gap:16px;padding:8px 0;border-bottom:1px solid var(--line);font-size:.85rem}
.ev-row span:first-child{color:var(--a);font-weight:700}
.map-img{max-width:100%;border-radius:14px;border:1px solid var(--line)}
.empty{border:2px dashed var(--a);border-radius:12px;padding:40px;text-align:center;color:var(--a);opacity:.7}
table.tbl{border-collapse:collapse;width:100%;font-size:.8rem}
table.tbl th,table.tbl td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line)}
table.tbl th{color:var(--a)}
.tbl-wrap{overflow-x:auto;max-height:55vh}
.report{white-space:pre-wrap;font-size:.85rem;line-height:1.7;max-height:70vh;overflow:auto}
select.form-control,.selectize-input{font-family:'Roboto Mono',monospace}
"

scroll_js <- "
(function(){
  function update(){var vh=innerHeight;document.querySelectorAll('.fade-item').forEach(function(el){
    var r=el.getBoundingClientRect(),o=r.top+r.height/2-vh/2,rg=o>0?.8:1,t=Math.min(Math.abs(o)/(vh*rg),1);
    el.style.opacity=.2+.8*(1-t*t);el.style.transform='translateY('+(Math.sign(o)*t*40)+'px)';});}
  var tk=false;function os(){if(!tk){tk=true;requestAnimationFrame(function(){update();tk=false;});}}
  addEventListener('scroll',os,{passive:true});addEventListener('resize',os);addEventListener('load',update);
  document.addEventListener('shiny:value',function(){setTimeout(update,50);});update();
})();
"

stat_card <- function(value, label, note = NULL, warn = FALSE) {
  div(class = if (warn) "stat-card warn" else "stat-card",
      div(class = "stat-value", value), div(class = "stat-label", label),
      if (!is.null(note)) div(class = "stat-note", note))
}
empty_box <- function(msg) div(class = "empty", msg)

# ---- UI ----
ui <- fluidPage(
  title = "Spatial Analysis of Cities",
  tags$head(tags$style(HTML(app_css)), tags$script(HTML(scroll_js))),

  div(class = "hero", div(class = "fade-item",
    h1(class = "hero-title", "Spatial Analysis of Cities: can a neural network tell monocentric from polycentric?"),
    div(class = "hero-sub", "Unsupervised learning on OpenStreetMap and GHSL data, checked against classical subcentre tests."),
    div(class = "hero-team", tags$b("Team"), "Your name \u00B7 Teammates \u00B7 Course / date")
  )),

  # 1. Idea
  div(class = "dash-section",
    div(class = "fade-item",
      h2(class = "section-heading", "The Idea"),
      p(class = "lead", "Cities are often called monocentric (one dominant centre) or polycentric (several). We rasterize each city into grids of POIs, buildings, roads and built-up area, train an autoencoder without labels, cluster what it learns, and compare the groups with classical tests."),
      div(class = "callout", p("The first goal is not a final answer. It is to narrow down broad trends so we know where to look closer."))
    ),
    div(class = "fade-item",
      h3(class = "sub-heading", "Pipeline"),
      p(class = "sub-note", "Extract (R) \u2192 Rasterize \u2192 Learn (autoencoder) \u2192 Cluster (k-means / HDBSCAN, UMAP) \u2192 Check against classical tests."),
      div(class = "stat-grid",
        stat_card("100 m", "Grid cell", "Local projection, one layer per feature"),
        stat_card("2 modes", "City and tile", "One ~40 km window per city, or many 3.2 km patches"),
        stat_card("2 tests", "Giuliano\u2013Small, McMillen", "Rule-based subcentre checks"))
    )
  ),

  # 2. Results overview
  div(class = "dash-section",
    div(class = "fade-item",
      h2(class = "section-heading", "Results Overview"),
      p(class = "lead", "Live from the latest run. Re-run the analysis and this page updates by itself."),
      uiOutput("status_msg")
    ),
    div(class = "fade-item", uiOutput("summary_cards")),
    div(class = "fade-item", div(class = "two",
      div(class = "plot-card", h4("City types"), plotOutput("plot_types", height = "320px")),
      div(class = "plot-card", h4("Confidence by type"), plotOutput("plot_conf", height = "320px"))
    )),
    div(class = "fade-item", uiOutput("size_block"))
  ),

  # 3. Map
  div(class = "dash-section",
    div(class = "fade-item",
      h2(class = "section-heading", "Where the Types Fall"),
      uiOutput("map_block"))
  ),

  # 4. City explorer
  div(class = "dash-section",
    div(class = "fade-item",
      h2(class = "section-heading", "City Explorer"),
      p(class = "sub-note", "Pick a city to see its type and all the evidence behind it."),
      uiOutput("city_picker"), uiOutput("city_evidence")
    ),
    div(class = "fade-item",
      h3(class = "sub-heading", "Browse by type"),
      uiOutput("type_picker"),
      div(class = "tbl-wrap", uiOutput("type_table")))
  ),

  # 5. Report + caveats
  div(class = "dash-section",
    div(class = "fade-item",
      h2(class = "section-heading", "Report and Caveats"),
      uiOutput("report_block"),
      div(class = "callout", h4("Read with care"),
        p("OSM coverage is uneven, POIs stand in for jobs, and very large GHSL urban centres can span a whole district, so read their type as a statement about the region. Labels from the literature are expectations, not facts.")))
  )
)

# ---- Server ----
server <- function(input, output, session) {

  raw <- reactiveFileReader(2000, session, csv_path, function(p) {
    if (file.exists(p)) read.csv(p, stringsAsFactors = FALSE, check.names = FALSE) else NULL
  })

  d <- reactive({
    df <- raw(); req(df, nrow(df) > 0)
    cols <- list(
      type = find_col(df, c("city_type", "type", "final_type", "classification", "class"), cfg$type),
      conf = find_col(df, c("confidence", "conf"), cfg$confidence),
      name = find_col(df, c("city", "name", "city_name", "urban_centre", "uc_name", "slug"), cfg$name),
      pop  = find_col(df, c("population", "pop", "pop_2020", "pop_2025", "total_pop"), cfg$population))
    list(df = df, cols = cols)
  })

  output$status_msg <- renderUI({
    if (is.null(raw())) empty_box(paste("No results yet. Expected file:", csv_path))
  })

  output$summary_cards <- renderUI({
    x <- d(); df <- x$df; k <- x$cols
    cards <- list(stat_card(comma(nrow(df)), "Cities classified"))
    if (!is.na(k$type)) {
      tt <- sort(table(df[[k$type]]), decreasing = TRUE)
      cards <- c(cards, lapply(seq_len(min(3, length(tt))), function(i)
        stat_card(comma(tt[[i]]), names(tt)[i], paste0(percent(tt[[i]] / nrow(df), 1), " of cities"),
                  warn = grepl("unclear", names(tt)[i], ignore.case = TRUE))))
    }
    if (!is.na(k$conf)) {
      hi <- mean(tolower(df[[k$conf]]) == "high", na.rm = TRUE)
      cards <- c(cards, list(stat_card(percent(hi, 1), "High confidence", "GHSL and OSM tests agree, well mapped")))
    }
    div(class = "stat-grid", cards)
  })

  output$plot_types <- renderPlot({
    x <- d(); k <- x$cols; req(!is.na(k$type))
    ct <- as.data.frame(table(type = x$df[[k$type]]))
    ggplot(ct, aes(reorder(type, Freq), Freq, fill = type)) +
      geom_col(show.legend = FALSE, width = .7) + coord_flip() +
      scale_fill_manual(values = rep(pal, length.out = nrow(ct))) +
      labs(x = NULL, y = "Cities") + theme_minimal(base_family = "mono", base_size = 13) +
      theme(panel.grid.major.y = element_blank())
  })

  output$plot_conf <- renderPlot({
    x <- d(); k <- x$cols; req(!is.na(k$type), !is.na(k$conf))
    ct <- as.data.frame(table(type = x$df[[k$type]], conf = x$df[[k$conf]]))
    ggplot(ct, aes(type, Freq, fill = conf)) + geom_col(width = .7) +
      scale_fill_manual(values = rep(pal, length.out = length(unique(ct$conf))), name = NULL) +
      labs(x = NULL, y = "Cities") + theme_minimal(base_family = "mono", base_size = 13) +
      theme(legend.position = "bottom", axis.text.x = element_text(angle = 20, hjust = 1))
  })

  output$size_block <- renderUI({
    k <- d()$cols
    if (is.na(k$type) || is.na(k$pop)) return(NULL)
    div(class = "plot-card", h4("Type mix by city size"), plotOutput("plot_size", height = "320px"))
  })
  output$plot_size <- renderPlot({
    x <- d(); k <- x$cols; req(!is.na(k$type), !is.na(k$pop))
    df <- data.frame(type = x$df[[k$type]], pop = suppressWarnings(as.numeric(x$df[[k$pop]])))
    df <- df[!is.na(df$pop), ]; req(nrow(df) > 0)
    df$size <- cut(df$pop, c(0, 5e5, 1e6, 5e6, Inf), labels = c("<0.5M", "0.5\u20131M", "1\u20135M", ">5M"))
    ggplot(df, aes(size, fill = type)) + geom_bar(position = "fill", width = .7) +
      scale_y_continuous(labels = percent) +
      scale_fill_manual(values = rep(pal, length.out = length(unique(df$type))), name = NULL) +
      labs(x = "Urban centre population", y = "Share of cities") +
      theme_minimal(base_family = "mono", base_size = 13) + theme(legend.position = "bottom")
  })

  output$map_block <- renderUI({
    if (file.exists(map_path)) {
      tagList(p(class = "sub-note", "Every city by type and by network group (city_types_map.png)."),
              tags$img(class = "map-img", src = "res/city_types_map.png", alt = "City types map"))
    } else empty_box(paste("Map not found:", map_path))
  })

  output$city_picker <- renderUI({
    x <- d(); k <- x$cols
    labs <- if (!is.na(k$name)) x$df[[k$name]] else seq_len(nrow(x$df))
    selectInput("city", NULL, choices = setNames(seq_len(nrow(x$df)), labs), width = "420px")
  })
  output$city_evidence <- renderUI({
    x <- d(); req(input$city)
    row <- x$df[as.integer(input$city), , drop = FALSE]
    div(class = "plot-card", lapply(names(row), function(n)
      div(class = "ev-row", span(n), span(as.character(row[[n]])))))
  })

  output$type_picker <- renderUI({
    x <- d(); k <- x$cols; req(!is.na(k$type))
    selectInput("type_sel", NULL, choices = sort(unique(x$df[[k$type]])), width = "420px")
  })
  output$type_table <- renderUI({
    x <- d(); k <- x$cols; req(input$type_sel, !is.na(k$type))
    df <- x$df[x$df[[k$type]] == input$type_sel, , drop = FALSE]
    if (!is.na(k$pop)) df <- df[order(-suppressWarnings(as.numeric(df[[k$pop]]))), , drop = FALSE]
    df <- head(df, 50)
    tags$table(class = "tbl",
      tags$tr(lapply(names(df), tags$th)),
      lapply(seq_len(nrow(df)), function(i) tags$tr(lapply(df[i, ], function(v) tags$td(as.character(v))))))
  })

  output$report_block <- renderUI({
    if (!file.exists(rep_path)) return(empty_box(paste("Report not found:", rep_path)))
    if (requireNamespace("markdown", quietly = TRUE))
      div(class = "plot-card report", HTML(markdown::markdownToHTML(rep_path, fragment.only = TRUE)))
    else div(class = "plot-card report", paste(readLines(rep_path, warn = FALSE), collapse = "\n"))
  })
}

shinyApp(ui, server)
