# FSOC PAT Simulator - Frontend

This is the Next.js frontend for the Free Space Optical Communication Point, Acquisition, and Tracking simulator.

## Getting Started

First, install dependencies:

```bash
npm install
```

Then, run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the dashboard.

## Features

- **Real-time Dashboard**: Displays PAT state, beacon position, tracking error, FPS, and gimbal angles
- **Virtual Camera View**: Visualizes the beacon and tracking crosshair
- **Disturbance Controls**: Sliders for turbulence, vibration, camera motion, and sensor noise
- **Control Buttons**: Start, pause, and reset the simulation
- **WebSocket Ready**: Prepared for real-time data streaming from the Python backend

## Architecture

```
Python Backend (WebSocket Server)
       ↓
WebSocket Connection
       ↓
Next.js Frontend
       ↓
React Dashboard Components
```

## Next Steps

1. Install dependencies: `npm install`
2. Run dev server: `npm run dev`
3. Implement WebSocket connection to backend
4. Add real-time frame streaming
5. Add telemetry graphs with Chart.js or similar
