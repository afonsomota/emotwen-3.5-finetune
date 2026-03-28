import type { Rating, Tag } from "../types";
import { ALL_TAGS } from "../types";

interface RatingBarProps {
  rating: Rating | null;
  tags: Tag[];
  onRate: (rating: Rating) => void;
  onToggleTag: (tag: Tag) => void;
  error: string | null;
  isEnd: boolean;
}

const TAG_LABELS: Record<Tag, string> = {
  "advice-leaking": "Advice (A)",
  "too-long": "Long (L)",
  "off-topic": "Off-topic (O)",
  "repetitive": "Repeat (R)",
};

export default function RatingBar({ rating, tags, onRate, onToggleTag, error, isEnd }: RatingBarProps) {
  return (
    <div className="ratingbar">
      <div className="ratingbar-row">
        <button
          className={`rate-btn rate-btn-bad${rating === "bad" ? " active" : ""}`}
          onClick={() => onRate("bad")}
        >
          Bad (1)
        </button>
        <button
          className={`rate-btn rate-btn-okay${rating === "okay" ? " active" : ""}`}
          onClick={() => onRate("okay")}
        >
          Okay (2)
        </button>
        <button
          className={`rate-btn rate-btn-good${rating === "good" ? " active" : ""}`}
          onClick={() => onRate("good")}
        >
          Good (3)
        </button>
        {isEnd && <span className="end-badge">End of list</span>}
      </div>
      <div className="ratingbar-row">
        {ALL_TAGS.map((tag) => (
          <button
            key={tag}
            className={`tag-chip${tags.includes(tag) ? " active" : ""}`}
            onClick={() => onToggleTag(tag)}
          >
            {TAG_LABELS[tag]}
          </button>
        ))}
      </div>
      <div className="ratingbar-row">
        <span className="shortcut-legend">
          1/2/3 rate &middot; a/l/o/r tags &middot; arrows nav &middot; s skip &middot; u unreviewed
        </span>
      </div>
      {error && <div className="error-toast">{error}</div>}
    </div>
  );
}
