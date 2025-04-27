import React from "react";
import { Routes, Route } from "react-router-dom";
import FashionWeb from "./Components/FashionWeb";

function App() {
  return (
    <div className="App">
      <Routes>
        <Route path="/" element={<FashionWeb />} />
      </Routes>
    </div>
  );
}

export default App;