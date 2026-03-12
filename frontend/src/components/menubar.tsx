import { useState, useEffect } from "react";

export default function Menubar() {
  const [dark, setDark] = useState(() =>
    document.documentElement.classList.contains("dark")
  );

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  return (
    <nav className="flex items-center justify-between px-6 py-3 bg-white dark:bg-gray-900 shadow">
      <span className="text-xl font-bold text-gray-900 dark:text-white">
        CancerVision
      </span>
      <button
        onClick={() => setDark(!dark)}
        className="px-3 py-1 rounded bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 text-sm"
      >
        {dark ? " Light" : "Dark"}
      </button>
    </nav>
  );
}
