import { useEffect, useRef } from "react";
import type { Message } from "../types";

interface ChatViewProps {
  messages: Message[];
  source: string;
  emotionLabel: string;
}

export default function ChatView({ messages, source, emotionLabel }: ChatViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = 0;
    }
  }, [messages]);

  const systemMsg = messages.find((m) => m.role === "system");
  const chatMessages = messages.filter((m) => m.role !== "system");

  return (
    <div className="chatview" ref={containerRef}>
      <div className="chat-meta">
        {source} &mdash; {emotionLabel}
      </div>
      {systemMsg && (
        <details className="chat-system">
          <summary>System prompt</summary>
          <pre>{systemMsg.content}</pre>
        </details>
      )}
      {chatMessages.length === 0 ? (
        <div className="chat-empty">(No conversation turns.)</div>
      ) : (
        chatMessages.map((msg, i) => (
          <div
            key={i}
            className={`bubble ${msg.role === "user" ? "bubble-user" : "bubble-assistant"}`}
          >
            {msg.content}
          </div>
        ))
      )}
    </div>
  );
}
