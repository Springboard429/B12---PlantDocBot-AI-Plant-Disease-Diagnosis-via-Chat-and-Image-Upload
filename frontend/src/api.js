const BASE_URL = "http://127.0.0.1:8000";

// ============================
// IMAGE API
// ============================
export async function predictImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${BASE_URL}/predict/image`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) throw new Error("Image API failed");

  return await res.json();
}

// ============================
// TEXT API
// ============================
export async function predictText(text) {
  const formData = new URLSearchParams();
  formData.append("text", text);

  const res = await fetch(`${BASE_URL}/predict/text`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) throw new Error("Text API failed");

  return await res.json();
}