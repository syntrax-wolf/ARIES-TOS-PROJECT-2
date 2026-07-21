import type { ButtonHTMLAttributes, ReactNode } from "react";
import clsx from "clsx";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "ghost";
  size?: "md" | "lg";
  children: ReactNode;
}

export function Button({ variant = "primary", size = "md", className, children, ...rest }: ButtonProps) {
  return (
    <button
      className={clsx(
        "inline-flex items-center justify-center gap-2 rounded-xl font-medium transition-all duration-150 select-none",
        "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand disabled:cursor-not-allowed disabled:opacity-40",
        size === "md" ? "px-5 py-2.5 text-[15px]" : "px-7 py-3.5 text-base",
        variant === "primary" &&
          "bg-brand text-white shadow-soft hover:bg-brand-dark active:scale-[0.98] disabled:hover:bg-brand",
        variant === "secondary" &&
          "bg-surface text-ink border border-border hover:border-ink-faint active:scale-[0.98]",
        variant === "ghost" && "text-ink-soft hover:bg-surface-soft active:scale-[0.98]",
        className
      )}
      {...rest}
    >
      {children}
    </button>
  );
}
