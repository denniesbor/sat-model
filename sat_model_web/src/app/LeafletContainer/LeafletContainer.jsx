import React, { useState, useEffect } from "react";
import DeckGL from "@deck.gl/react";
import { Map } from "react-map-gl";
import { PolygonLayer } from "@deck.gl/layers";

const INITIAL_VIEW_STATE = {
  longitude: 0,
  latitude: 0,
  zoom: 1,
  pitch: 0,
  bearing: 0,
};

const DensityMap = () => {
  const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);
  const [data, setData] = useState([]);
  const [densityRange, setDensityRange] = useState({ min: 0, max: 0 });

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch("/data/density_grid.json");
        const result = await response.json();

        const polygons = [];
        let minDensity = Infinity;
        let maxDensity = -Infinity;

        // Create polygons for each grid cell
        for (let i = 0; i < result.lons.length - 1; i++) {
          for (let j = 0; j < result.lats.length - 1; j++) {
            if (result.densities?.[i]?.[j]?.[0]) {
              const density = result.densities[i][j][0];
              minDensity = Math.min(minDensity, density);
              maxDensity = Math.max(maxDensity, density);

              polygons.push({
                polygon: [
                  [
                    [result.lons[i], result.lats[j]],
                    [result.lons[i + 1], result.lats[j]],
                    [result.lons[i + 1], result.lats[j + 1]],
                    [result.lons[i], result.lats[j + 1]],
                    [result.lons[i], result.lats[j]],
                  ],
                ],
                density: density,
              });
            }
          }
        }

        console.log("Density range:", { minDensity, maxDensity });
        setDensityRange({ min: minDensity, max: maxDensity });
        setData(polygons);
      } catch (error) {
        console.error("Error:", error);
      }
    };

    fetchData();
  }, []);

  const layers = [
    new PolygonLayer({
      id: "density-layer",
      data,
      pickable: true,
      stroked: false,
      filled: true,
      extruded: false,
      getPolygon: (d) => d.polygon,
      getFillColor: (d) => {
        const normalized =
          (d.density - densityRange.min) /
          (densityRange.max - densityRange.min);
        const alpha = Math.floor(normalized * 255);
        return [255, 255, 0, alpha]; // Yellow with varying transparency
      },
      opacity: 1,
    }),
  ];

  return (
    <div className="w-[800px] h-[600px]">
      <DeckGL
        initialViewState={INITIAL_VIEW_STATE}
        viewState={viewState}
        onViewStateChange={({ viewState }) => setViewState(viewState)}
        controller={true}
        layers={layers}
      >
        <Map
          mapStyle="mapbox://styles/mapbox/dark-v9"
          mapboxAccessToken={import.meta.env.VITE_APP_MAPBOX_TOKEN}
        />
      </DeckGL>
    </div>
  );
};

export default DensityMap;
