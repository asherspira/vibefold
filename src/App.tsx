import React, { useState } from 'react';
import { WormholeTab } from './components/WormholeTab';
import { Orbit } from 'lucide-react';

export default function App() {
  const [wormholeUiVisible, setWormholeUiVisible] = useState(true);
  const hideNav = !wormholeUiVisible;

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-[#030712] flex flex-col font-sans select-none">
      {/* Top Header Bar */}
      <header
        className={`absolute top-0 left-0 right-0 z-30 flex items-center justify-between gap-2 overflow-hidden px-3 py-2 bg-slate-950/85 backdrop-blur-md border-b border-slate-800/80 shadow-lg sm:px-4 transition-all duration-300 ${
          hideNav ? '-translate-y-full opacity-0 pointer-events-none' : 'translate-y-0 opacity-100'
        }`}
      >
        {/* Brand / Logo */}
        <div className="flex min-w-0 shrink items-center gap-2.5">
          <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-gradient-to-br from-fuchsia-500/20 to-purple-600/30 border border-fuchsia-500/40 text-fuchsia-300">
            <Orbit className="w-4 h-4 text-fuchsia-300" />
          </div>
          <div className="flex min-w-0 flex-col">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-100 flex items-center gap-1.5">
              Gravitational Wormhole
              <span className="hidden sm:inline-block text-[9px] font-mono font-semibold px-1.5 py-0.2 rounded bg-fuchsia-950/70 border border-fuchsia-500/40 text-fuchsia-300">
                3D Raytracer
              </span>
            </span>
            <span className="text-[10px] text-slate-400">
              Gravitational Wormhole Ray Deflection
            </span>
          </div>
        </div>

        {/* Mode Tag */}
        <div className="hidden sm:flex items-center gap-2 text-xs">
          <span className="w-2 h-2 rounded-full bg-fuchsia-400 animate-pulse" />
          <span className="text-[11px] font-mono text-slate-400">Osipov shader 4dccRr</span>
        </div>
      </header>

      {/* Main Viewport */}
      <main
        className={`w-full h-full relative transition-all duration-300 ${
          hideNav ? 'pt-0' : 'pt-12'
        }`}
      >
        <WormholeTab
          uiVisible={wormholeUiVisible}
          onToggleUi={() => setWormholeUiVisible((prev) => !prev)}
        />
      </main>
    </div>
  );
}
