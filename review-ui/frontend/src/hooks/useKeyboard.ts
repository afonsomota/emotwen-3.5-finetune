import { useEffect } from "react";

export default function useKeyboard(handlers: Record<string, () => void>): void {
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      const target = e.target as HTMLElement;
      const tag = target.tagName.toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select") {
        return;
      }

      const key =
        e.key === "ArrowLeft" || e.key === "ArrowRight" ? e.key : e.key.toLowerCase();

      if (key in handlers) {
        e.preventDefault();
        handlers[key]();
      }
    }

    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [handlers]);
}
