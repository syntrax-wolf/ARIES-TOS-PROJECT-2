export default function Navbar() {
  return (
    <nav className="w-full h-16 bg-slate-900 border-b border-slate-800 flex items-center justify-between px-8">
      <div>
        <h1 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 via-purple-500 to-pink-500 bg-clip-text text-transparent">
          LogiScope
        </h1>
      </div>

      <div className="text-slate-400 text-sm">
        Interactive Logistic Regression Visualizer
      </div>
    </nav>
  );
}