import React, { useEffect, useState } from "react";
import { ImageryLayer, IonImageryProvider, Rectangle, Color } from "cesium";

const NightShading = ({ clock, layers }) => {
  const [blackMarbleLayer, setBlackMarbleLayer] = useState(null);

  useEffect(() => {
    if (!layers) return;

    const initBlackMarble = async () => {
      try {
        // Create Black Marble layer
        const blackMarble = await ImageryLayer.fromProviderAsync(
          IonImageryProvider.fromAssetId(3812)
        );

        // Configure layer
        blackMarble.alpha = 0.8;
        blackMarble.brightness = 1.5;
        blackMarble.contrast = 1.2;
        blackMarble.show = true;

        setBlackMarbleLayer(blackMarble);
        layers.add(blackMarble);
      } catch (error) {
        console.error("Error initializing Black Marble layer:", error);
      }
    };

    initBlackMarble();

    return () => {
      if (blackMarbleLayer && layers) {
        layers.remove(blackMarbleLayer);
      }
    };
  }, [layers]);

  const loadNightData = async () => {
    try {
      const response = await fetch("/data/night_data.json");
      const data = await response.json();

      if (blackMarbleLayer) {
        // Create rectangle from night boundary
        const lons = data.nightBoundary.map((p) => p[0]);
        const lats = data.nightBoundary.map((p) => p[1]);

        const rectangle = Rectangle.fromDegrees(
          Math.min(...lons),
          Math.min(...lats),
          Math.max(...lons),
          Math.max(...lats)
        );

        // Apply rectangle as cutout mask for Black Marble
        blackMarbleLayer.cutoutRectangle = rectangle;
      }
    } catch (error) {
      console.error("Error loading night data:", error);
    }
  };

  useEffect(() => {
    if (clock && blackMarbleLayer) {
      const updateNightRegion = () => {
        loadNightData();
      };

      const interval = setInterval(updateNightRegion, 60000);
      updateNightRegion();

      return () => clearInterval(interval);
    }
  }, [clock, blackMarbleLayer]);

  return null;
};

export default NightShading;
