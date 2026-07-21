import { useEffect } from "react";
import axios from "axios";
import Dashboard from "./pages/Dashboard";

function App() {

  useEffect(() => {

    axios.get("http://127.0.0.1:5000/")
      .then((response) => {
        console.log(response.data);
      })
      .catch((error) => {
        console.log(error);
      });

  }, []);

  return <Dashboard />;
}

export default App;