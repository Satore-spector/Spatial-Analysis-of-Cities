import "./style.css";
import { heroGrid, carousel, scrollFade } from "./ui";
import { loadCities } from "./data";
import { renderResults, renderError } from "./results";

heroGrid();
carousel();
scrollFade();
loadCities().then(renderResults).catch(renderError);
