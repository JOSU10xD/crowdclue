// frontend/src/api.js
const BASE = import.meta.env.VITE_API_BASE || 'http://localhost:3000'
const ADMIN_SECRET = import.meta.env.VITE_ADMIN_SECRET || ''  // optional (demo convenience)

function _json(res) {
  return res.json().then(body => ({ ok: res.ok, status: res.status, body }))
}

function getToken() {
  return localStorage.getItem('token')
}
function setToken(token) {
  if (token) localStorage.setItem('token', token)
  else localStorage.removeItem('token')
}
function clearToken() {
  localStorage.removeItem('token')
}

async function _fetch(path, opts = {}) {
  const headers = Object.assign({}, opts.headers || {})
  const token = getToken()
  if (token) headers['Authorization'] = `Bearer ${token}`
  // include admin secret for admin endpoints if provided (demo convenience only)
  if (ADMIN_SECRET && (path.startsWith('/admin') || path.startsWith('/admin/'))) {
    headers['x-admin-secret'] = ADMIN_SECRET
  }
  const res = await fetch(BASE + path, { ...opts, headers })
  return _json(res)
}

export default {
  async login(name, password) {
    return _fetch('/api/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, password }),
    })
  },
  async register(name, password) {
    return _fetch('/api/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, password }),
    })
  },
  async listImages() { return _fetch('/api/images') },
  async getImage(id) { return _fetch(`/api/images/${id}`) },
  async postClue(imageId, text) {
    return _fetch(`/api/images/${imageId}/clues`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    })
  },
  async adminUpload(formData) {
    // admin secret may be inserted automatically by _fetch based on ADMIN_SECRET
    const token = getToken()
    const headers = token ? { Authorization: `Bearer ${token}` } : {}
    // don't set Content-Type; browser will set multipart boundary
    const res = await fetch(BASE + '/admin/images', {
      method: 'POST',
      headers,
      body: formData,
    })
    return _json(res)
  },
  async adminListClues() { return _fetch('/admin/clues?status=pending') },
  async adminAcceptClue(id) { return _fetch(`/admin/clues/${id}/accept`, { method: 'POST' }) },
  async scoreboard() { return _fetch('/api/scoreboard') },
}

export { setToken, getToken, clearToken }
