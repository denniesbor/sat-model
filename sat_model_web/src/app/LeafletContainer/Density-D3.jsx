import React, { useEffect } from "react";
import L from "leaflet";
import * as d3 from "d3";
import { useMap } from "react-leaflet";

const DensityOverlay = ({ densityData }) => {
  const map = useMap();
  
  // console.log("DensityOverlay", densityData);
  useEffect(() => {
    if (!densityData) return;

    // Create SVG layer
    const svg = d3
      .select(map.getPanes().overlayPane)
      .append("svg")
      .attr("class", "leaflet-zoom-hide");
    const g = svg.append("g");

    // Color scale
    const values = densityData.values.flat();
    const colorScale = d3
      .scaleSequential(d3.interpolateViridis)
      .domain([d3.min(values), d3.max(values)]);

    // Setup projection
    const transform = d3.geoTransform({ point: projectPoint });
    const path = d3.geoPath().projection(transform);

    // Generate grid features
    const gridFeatures = densityData.lats.slice(0, -1).flatMap((lat, i) =>
      densityData.lons.slice(0, -1).map((lon, j) => ({
        type: "Feature",
        geometry: {
          type: "Polygon",
          coordinates: [
            [
              [lon, lat],
              [densityData.lons[j + 1], lat],
              [densityData.lons[j + 1], densityData.lats[i + 1]],
              [lon, densityData.lats[i + 1]],
              [lon, lat],
            ],
          ],
        },
        properties: { density: densityData.values[i][j] },
      }))
    );

    // Render cells
    g.selectAll("path")
      .data(gridFeatures)
      .enter()
      .append("path")
      .attr("d", path)
      .style("fill", (d) => colorScale(d.properties.density))
      .style("opacity", 0.7);

    // Update functions
    function reset() {
      const bounds = path.bounds({
        type: "FeatureCollection",
        features: gridFeatures,
      });
      const [topLeft, bottomRight] = bounds;

      svg
        .attr("width", bottomRight[0] - topLeft[0])
        .attr("height", bottomRight[1] - topLeft[1])
        .style("left", `${topLeft[0]}px`)
        .style("top", `${topLeft[1]}px`);

      g.attr("transform", `translate(${-topLeft[0]},${-topLeft[1]})`);
    }

    function projectPoint(x, y) {
      const point = map.latLngToLayerPoint(new L.LatLng(y, x));
      this.stream.point(point.x, point.y);
    }

    // Event handlers
    map.on("zoomend moveend", reset);
    reset();

    return () => {
      svg.remove();
      map.off("zoomend moveend", reset);
    };
  }, [map, densityData]);

  return null;
};

export default DensityOverlay;
