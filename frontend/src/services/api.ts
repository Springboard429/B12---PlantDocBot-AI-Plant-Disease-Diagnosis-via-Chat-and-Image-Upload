// handles api calls

const BASE_URL = "http://127.0.0.1:8000";

export const predictImage = async (file: File) => {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${BASE_URL}/predict-image`, {
    method: "POST",
    body: formData,
  });

  return res.json();
};

export const predictText = async (text: string) => {
  const res = await fetch(`${BASE_URL}/predict-text`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text }),
  });

  return res.json();
};

export const predictCombined = async (text: string, file: File | null) => {
  const formData = new FormData();

  if (text) formData.append("text", text);
  if (file) formData.append("file", file);

  const res = await fetch(`${BASE_URL}/predict-combined`, {
    method: "POST",
    body: formData,
  });

  return res.json();
};