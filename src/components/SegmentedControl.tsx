import clsx from "clsx";

interface Option<T extends string> {
  value: T;
  label: string;
}

interface SegmentedControlProps<T extends string> {
  label?: string;
  options: Option<T>[];
  value: T;
  onChange: (value: T) => void;
}

export function SegmentedControl<T extends string>({ label, options, value, onChange }: SegmentedControlProps<T>) {
  return (
    <div>
      {label && <div className="mb-2 text-sm font-medium text-ink">{label}</div>}
      <div className="inline-flex w-full rounded-xl bg-surface-soft p-1">
        {options.map((opt) => (
          <button
            key={opt.value}
            type="button"
            onClick={() => onChange(opt.value)}
            className={clsx(
              "flex-1 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors",
              value === opt.value ? "bg-surface text-brand-dark shadow-card" : "text-ink-soft hover:text-ink"
            )}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
}
