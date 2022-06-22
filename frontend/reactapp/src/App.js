import Home from './components/Home/home';
import Videos from './components/Videos/Videos';
import {
  BrowserRouter as Router,
  Routes,
  Route,
} from "react-router-dom";

function App() {
  return (
    <div>
        <Router>
            <Routes>
               <Route path="/" element={<Home />} />
               <Route path="/videos" element={<Videos />} />
            </Routes>
        </Router>
    </div>
  );
}

export default App;
