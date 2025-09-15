// frontend/src/utils/jwt.js
export function parseJwt(token) {
  if (!token) return null;
  try {
    const base64 = token.split('.')[1];
    if (!base64) return null;
    const s = base64.replace(/-/g, '+').replace(/_/g, '/');
    const jsonPayload = decodeURIComponent(
      atob(s)
        .split('')
        .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join('')
    );
    return JSON.parse(jsonPayload);
  } catch (err) {
    return null;
  }
}
