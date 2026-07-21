import clsx from "clsx";

const STEPS = ["Algorithm", "Configure", "Grow", "Results"];

export function Logomark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} fill="none">
      <path
        d="M12 21V13"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
      />
      <path
        d="M12 13C12 13 5 12.5 5 6.5C5 6.5 12 5 12 13Z"
        fill="currentColor"
        opacity="0.85"
      />
      <path
        d="M12 13C12 13 19 12.5 19 6.5C19 6.5 12 5 12 13Z"
        fill="currentColor"
      />
    </svg>
  );
}

export function StepHeader({ step }: { step: number }) {
  return (
    <header className="flex items-center justify-between px-8 py-6 sm:px-12">
      <div className="flex items-center gap-2 text-brand-dark">
        <Logomark className="h-6 w-6" />
        <span className="font-serif text-lg font-semibold tracking-tight">Sylva</span>
      </div>

      <ol className="flex items-center gap-2 sm:gap-3">
        {STEPS.map((label, i) => {
          const idx = i + 1;
          const state = idx === step ? "current" : idx < step ? "done" : "upcoming";
          return (
            <li key={label} className="flex items-center gap-2 sm:gap-3">
              <div className="flex items-center gap-2">
                <span
                  className={clsx(
                    "flex h-6 w-6 items-center justify-center rounded-full text-[12px] font-medium transition-colors",
                    state === "current" && "bg-brand text-white",
                    state === "done" && "bg-brand-soft text-brand-dark",
                    state === "upcoming" && "bg-surface-soft text-ink-faint"
                  )}
                >
                  {idx}
                </span>
                <span
                  className={clsx(
                    "hidden text-sm font-medium sm:inline",
                    state === "current" ? "text-ink" : "text-ink-faint"
                  )}
                >
                  {label}
                </span>
              </div>
              {idx !== STEPS.length && <span className="h-px w-4 bg-border sm:w-6" />}
            </li>
          );
        })}
      </ol>
    </header>
  );
}
