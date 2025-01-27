import React, { useEffect, useState, useCallback, useRef } from "react";
import { Viewer } from "resium";
import {
  Ion,
  createWorldTerrainAsync,
  SampledPositionProperty,
  Cartesian3,
  JulianDate,
  IonImageryProvider,
} from "cesium";
import { createSpinningClock } from "./SpinningClock";
import TLEPropagator from "./TLEPropagator";
import NightShading from "./NightShading";

Ion.defaultAccessToken = import.meta.env.VITE_CESIUM_ION_ACCESS_TOKEN;

const BATCH_SIZE = 50;
const BATCH_INTERVAL = 100;

const CesiumContainer = () => {
  const [terrainProvider, setTerrainProvider] = useState(null);
  const [satellites, setSatellites] = useState({});
  const [pathData, setPathData] = useState(null);
  const [clock, setClock] = useState(null);
  const [viewer, setViewer] = useState(null);
  const [viewerReady, setViewerReady] = useState(false);
  const [loading, setLoading] = useState({ current: 0, total: 0 });
  const [selectedSatellite, setSelectedSatellite] = useState(null);
  const [hoveredSatellite, setHoveredSatellite] = useState(null);

  const queueRef = useRef([]);
  const processingRef = useRef(false);

  const handleViewerRef = useCallback((viewerInstance) => {
    if (viewerInstance?.cesiumElement) {
      const cesiumViewer = viewerInstance.cesiumElement;
      setViewer(cesiumViewer);

      // Wait for the scene to be ready
      cesiumViewer.scene.globe.tileLoadProgressEvent.addEventListener(
        (remaining) => {
          if (remaining === 0) {
            setViewerReady(true);
          }
        }
      );
    }
  }, []);

  const handleSatelliteSelect = useCallback((id) => {
    setSelectedSatellite((prevId) => (prevId === id ? null : id));
  }, []);

  const handleSatelliteHover = useCallback((id) => {
    setHoveredSatellite(id);
  }, []);

  const createSatelliteProperty = useCallback((satData, metadata = null) => {
    const property = new SampledPositionProperty();

    if (metadata) {
      const startTime = new Date(metadata.start_time);
      for (let i = 0; i < metadata.n_steps; i++) {
        const time = new Date(
          startTime.getTime() + i * metadata.time_step * 1000
        );
        property.addSample(
          JulianDate.fromDate(time),
          Cartesian3.fromDegrees(
            satData.lon[i],
            satData.lat[i],
            satData.alt[i] * 1000
          )
        );
      }
    } else {
      satData.times.forEach((time, i) => {
        property.addSample(
          JulianDate.fromDate(new Date(time)),
          Cartesian3.fromDegrees(
            satData.lon?.[i] || satData.longitudes[i],
            satData.lat?.[i] || satData.latitudes[i],
            (satData.alt?.[i] || satData.altitudes[i]) * 1000
          )
        );
      });
    }
    return property;
  }, []);

  const processBatch = useCallback(() => {
    if (processingRef.current || queueRef.current.length === 0) return;

    processingRef.current = true;
    const batch = queueRef.current.splice(0, BATCH_SIZE);

    const newSatellites = {};
    batch.forEach(([id, satData, metadata]) => {
      try {
        newSatellites[id] = createSatelliteProperty(satData, metadata);
      } catch (error) {
        console.warn(`Failed to process satellite ${id}:`, error);
      }
    });

    setSatellites((prev) => ({ ...prev, ...newSatellites }));
    setLoading((prev) => ({
      ...prev,
      current: prev.total - queueRef.current.length,
    }));

    processingRef.current = false;

    if (queueRef.current.length > 0) {
      setTimeout(processBatch, BATCH_INTERVAL);
    }
  }, [createSatelliteProperty]);

  useEffect(() => {
    const loadData = async () => {
      try {
        const terrainPromise = createWorldTerrainAsync();

        const response = await fetch("/data/all_sats_optimized.json");
        const data = await response.json();

        const { metadata, satellites } = data.metadata
          ? data
          : { metadata: null, satellites: data };
        setPathData(satellites);

        const now = JulianDate.now();
        const newClock = createSpinningClock(now);
        setClock(newClock);

        const entries = Object.entries(satellites);
        queueRef.current = entries.map(([id, satData]) => [
          id,
          satData,
          metadata,
        ]);
        setLoading({ current: 0, total: entries.length });

        processBatch();

        setTerrainProvider(await terrainPromise);
      } catch (error) {
        console.error("Error loading data:", error);
      }
    };

    loadData();
  }, [processBatch]);

  if (!terrainProvider) return <div>Loading terrain...</div>;

  return (
    <>
      <Viewer
        full
        clockViewModel={clock}
        terrainProvider={terrainProvider}
        enableLighting={true}
        ref={handleViewerRef}
        imageryProvider={IonImageryProvider.fromAssetId(3)}
      >
        {viewer && viewerReady && (
          <NightShading clock={clock} layers={viewer.scene.imageryLayers} />
        )}

        {Object.entries(satellites).map(([id, positionProperty]) => (
          <TLEPropagator
            key={id}
            id={id}
            positionProperty={positionProperty}
            satData={pathData[id]}
            isSelected={selectedSatellite === id}
            showPath={selectedSatellite === id || hoveredSatellite === id}
            onSelect={handleSatelliteSelect}
            onHover={handleSatelliteHover}
          />
        ))}
      </Viewer>

      {loading.current < loading.total && (
        <div
          style={{
            position: "absolute",
            top: 10,
            right: 10,
            background: "rgba(0,0,0,0.7)",
            color: "white",
            padding: "8px 12px",
            borderRadius: "4px",
            fontSize: "14px",
          }}
        >
          Loading satellites:{" "}
          {Math.round((loading.current / loading.total) * 100)}%
        </div>
      )}

      {selectedSatellite && pathData && (
        <div
          style={{
            position: "absolute",
            top: 10,
            left: 10,
            background: "rgba(0,0,0,0.7)",
            color: "white",
            padding: "12px",
            borderRadius: "4px",
            maxWidth: "300px",
          }}
        >
          <h3>{pathData[selectedSatellite].name}</h3>
          <button
            onClick={() => setSelectedSatellite(null)}
            style={{
              position: "absolute",
              top: 5,
              right: 5,
              background: "none",
              border: "none",
              color: "white",
              cursor: "pointer",
            }}
          >
            ×
          </button>
        </div>
      )}
    </>
  );
};

export default CesiumContainer;
