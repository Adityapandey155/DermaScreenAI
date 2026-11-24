import { DISCLAIMER } from "@/ml/conditions";

export default function Disclaimer({ className = "" }: { className?: string }) {
  return (
    <aside
      role="note"
      className={`flex gap-3 rounded-2xl border border-amber-200 bg-amber-50 p-4 text-sm leading-relaxed text-amber-900 ${className}`}
    >
      <svg className="mt-0.5 h-5 w-5 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0zM12 9v4M12 17h.01" />
      </svg>
      <p>
        <strong>Medical disclaimer. </strong>
        {DISCLAIMER}
      </p>
    </aside>
  );
}
