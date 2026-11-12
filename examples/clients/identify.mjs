// Node 20+ (server side, built-in fetch/FormData):
//   FACE_API_URL=https://faces.example.org FACE_API_KEY_FILE=client.key node identify.mjs photo.jpg
import { readFile } from "node:fs/promises";

const url = process.env.FACE_API_URL.replace(/\/$/, "");
const key = (await readFile(process.env.FACE_API_KEY_FILE, "utf8")).trim();

export async function identify(path, candidates = 3) {
  for (let attempt = 1; ; attempt++) {
    const form = new FormData();
    form.append("image", new Blob([await readFile(path)]), "photo.jpg");
    const res = await fetch(`${url}/v1/identify?candidates=${candidates}`, {
      method: "POST",
      headers: { Authorization: `Bearer ${key}` },
      body: form,
      signal: AbortSignal.timeout(120_000),
    });
    if ([429, 502, 503].includes(res.status) && attempt < 3) {
      await new Promise((r) => setTimeout(r, 1000 * Number(res.headers.get("retry-after") ?? 5)));
      continue;
    }
    if (!res.ok) throw new Error(`face api ${res.status}: ${await res.text()}`);
    return res.json();
  }
}

const result = await identify(process.argv[2]);
for (const f of result.faces) console.log(f.box, f.name ?? "unknown", f.distance);
