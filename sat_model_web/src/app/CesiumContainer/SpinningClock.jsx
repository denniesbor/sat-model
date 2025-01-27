import {
  Clock,
  ClockRange,
  ClockStep,
  JulianDate,
  ClockViewModel,
} from "cesium";

export const createSpinningClock = () => {
  const now = JulianDate.now(); // Get the current date and time

  return new ClockViewModel(
    new Clock({
      startTime: JulianDate.addHours(now, -12, new JulianDate()), // Start 12 hours ago
      currentTime: now, // Start the simulation at the current time
      stopTime: JulianDate.addHours(now, 12, new JulianDate()), // Stop 12 hours into the future
      clockRange: ClockRange.LOOP_STOP, // The clock loops when it reaches `stopTime`
      clockStep: ClockStep.SYSTEM_CLOCK_MULTIPLIER, // Progresses in real-time or faster
      multiplier: 1, // Real-time progression (1 second in simulation = 1 real second)
      shouldAnimate: true, // Enables smooth spinning
    })
  );
};
