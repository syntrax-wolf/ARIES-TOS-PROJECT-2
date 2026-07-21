import type { HTMLAttributes, ReactNode } from "react";
import clsx from "clsx";

export function Card({ className, children, ...rest }: HTMLAttributes<HTMLDivElement> & { children: ReactNode }) {
  return (
    <div
      className={clsx("rounded-2xl border border-border bg-surface shadow-card", className)}
      {...rest}
    >
      {children}
    </div>
  );
}
