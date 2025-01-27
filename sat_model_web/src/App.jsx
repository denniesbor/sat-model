import { useState } from "react";
import "./App.css";
import { CesiumContainer } from "./app";
import { LeafletContainer } from "./app";

function App() {
  return (
    <div className="app">
      <div className="app__container">
        {/* <div className="app__leaflet">
      <LeafletContainer />
    </div> */}
        <div className="app__cesium">
          <CesiumContainer />
        </div>
      </div>
    </div>
  );
}

export default App;
