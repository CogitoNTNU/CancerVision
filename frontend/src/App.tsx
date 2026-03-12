import Menubar from "./components/menubar";

function App() {
  return (
    <div className="min-h-screen bg-white dark:bg-gray-950 text-gray-900 dark:text-gray-100">
      <Menubar />
      <main className="p-6">
        <p>Velkommen til CancerVision</p>
      </main>
    </div>
  );
}

export default App
