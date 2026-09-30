// Popover.tsx — a small click-outside/Escape-aware dropdown. Render it next
// to its trigger inside a positioned ancestor (.pop-anchor). place="down"
// floats below the trigger (default); "up" floats above it, for controls
// anchored near the bottom edge of the window.

import { useEffect, useRef, type ReactNode } from "react";

interface Props {
  open: boolean;
  onClose: () => void;
  align?: "left" | "right";
  place?: "down" | "up";
  className?: string;
  children: ReactNode;
}

export function Popover({ open, onClose, align = "left", place = "down", className, children }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div
      ref={ref}
      className={
        "popover" +
        (align === "right" ? " align-right" : "") +
        (place === "up" ? " place-up" : "") +
        (className ? ` ${className}` : "")
      }
    >
      {children}
    </div>
  );
}

/** One clickable row in a popover menu. */
export function PopoverItem({
  label,
  hint,
  selected,
  onClick,
}: {
  label: string;
  hint?: string;
  selected?: boolean;
  onClick: () => void;
}) {
  return (
    <button className={"popover-item" + (selected ? " selected" : "")} onClick={onClick}>
      <span>{label}</span>
      {hint && <span className="popover-hint">{hint}</span>}
      {selected && <span className="popover-check">✓</span>}
    </button>
  );
}
