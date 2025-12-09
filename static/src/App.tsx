import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Landing from "./Landing";
import { ComposerPage } from "./ComposerPage";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/composer" element={<ComposerPage />} />
      </Routes>
    </Router>
  );
}

export default App;
