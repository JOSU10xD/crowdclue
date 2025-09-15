// frontend/src/App.jsx
import React, { useEffect, useState } from 'react'
import api, { setToken, getToken, clearToken } from './api'
import { parseJwt } from './utils/jwt'

/* --- small components inline for simplicity --- */

function Header({ user, onLogout }) {
  return (
    <div style={{display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:12}}>
      <div><strong>CrowdClue</strong></div>
      <div>
        {user ? (
          <>
            <span style={{marginRight:8}}>Hello {user.name}</span>
            <button onClick={onLogout}>Logout</button>
          </>
        ) : <span>Not logged in</span>}
      </div>
    </div>
  )
}

function Login({ onLogin }) {
  const [name,setName] = useState('')
  const [pass,setPass] = useState('')
  const [msg,setMsg] = useState(null)
  async function doLogin(e){
    e.preventDefault()
    const r = await api.login(name, pass)
    if (r.ok) {
      const token = r.body.token
      setToken(token)
      const user = parseJwt(token)
      onLogin(user)
    } else {
      setMsg(JSON.stringify(r.body))
    }
  }
  return (
    <div style={{border:'1px solid #ddd', padding:12, marginBottom:8}}>
      <h3>Login</h3>
      <form onSubmit={doLogin}>
        <input placeholder="username" value={name} onChange={e=>setName(e.target.value)} />
        <input placeholder="password" value={pass} onChange={e=>setPass(e.target.value)} type="password" />
        <div style={{marginTop:8}}><button>Login</button></div>
      </form>
      {msg && <div style={{color:'red'}}>{msg}</div>}
    </div>
  )
}

function Register() {
  const [name,setName]=useState('')
  const [pass,setPass]=useState('')
  const [msg,setMsg]=useState(null)
  async function doReg(e){
    e.preventDefault()
    const r = await api.register(name, pass)
    setMsg(JSON.stringify(r.body))
  }
  return (
    <div style={{border:'1px solid #ddd', padding:12, marginBottom:8}}>
      <h3>Register</h3>
      <form onSubmit={doReg}>
        <input placeholder="username" value={name} onChange={e=>setName(e.target.value)} />
        <input placeholder="password" value={pass} onChange={e=>setPass(e.target.value)} type="password" />
        <div style={{marginTop:8}}><button>Register</button></div>
      </form>
      {msg && <div style={{marginTop:8}}>{msg}</div>}
    </div>
  )
}

function ImageList({ onOpen }) {
  const [images,setImages] = useState([])
  useEffect(()=>{ api.listImages().then(r=>{ if(r.ok) setImages(r.body) }) },[])
  return (
    <div style={{border:'1px solid #ddd', padding:12, marginBottom:8}}>
      <h3>Images</h3>
      {images.length===0 ? <div>No images</div> : (
        <ul>
          {images.map(img => (
            <li key={img.id} style={{marginBottom:8}}>
              <a href="#" onClick={(e)=>{ e.preventDefault(); onOpen(img.id) }}>{img.title || img.filename}</a>
              <div style={{fontSize:12,color:'#666'}}>{img.uploaded_at}</div>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function ImageDetail({ id, user, onBack }) {
  const [data,setData] = useState(null)
  const [text, setText] = useState('')
  const [msg, setMsg] = useState(null)
  useEffect(()=>{ api.getImage(id).then(r=>{ if(r.ok) setData(r.body) }) },[id])
  if (!data) return <div style={{border:'1px solid #ddd', padding:12}}>Loading…</div>
  const { image, clues } = data
  async function doClue(e){
    e.preventDefault()
    const r = await api.postClue(id, text)
    setMsg(JSON.stringify(r.body))
    if (r.ok) {
      setText('')
      api.getImage(id).then(rr=>{ if(rr.ok) setData(rr.body) })
    }
  }
  return (
    <div style={{border:'1px solid #ddd', padding:12, marginBottom:8}}>
      <button onClick={onBack}>← Back</button>
      <h3>{image.title}</h3>
      <img src={image.url} alt={image.title} style={{maxWidth:'100%'}} />
      <h4>Clues</h4>
      {clues.length===0 ? <div>No clues yet</div> : (
        <ul>{clues.map(c=>(<li key={c.id}><strong>{c.contributor}</strong>: {c.text} <em>({c.status})</em></li>))}</ul>
      )}
      {user ? (
        <form onSubmit={doClue}>
          <h4>Add clue</h4>
          <textarea value={text} onChange={e=>setText(e.target.value)} style={{width:'100%'}} />
          <div style={{marginTop:6}}><button>Submit clue</button></div>
          {msg && <div style={{marginTop:8}}>{msg}</div>}
        </form>
      ) : <div>Login to add clue</div>}
    </div>
  )
}

function AdminPanel({ token }) {
  // minimal admin UI: upload + list pending clues
  const [file, setFile] = useState(null)
  const [title, setTitle] = useState('')
  const [msg, setMsg] = useState(null)
  const [clues, setClues] = useState([])

  async function reloadClues() {
    const r = await api.adminListClues()
    if (r.ok) setClues(r.body)
  }
  useEffect(()=>{ reloadClues() },[])

  async function doUpload(e){
    e.preventDefault()
    if(!file) return setMsg('select file')
    const fd = new FormData()
    fd.append('image', file)
    fd.append('title', title || 'upload')
    const res = await api.adminUpload(fd)
    setMsg(JSON.stringify(res))
    if (res.ok) {
      reloadClues()
    }
  }

  async function accept(id){
    await api.adminAcceptClue(id)
    setClues(clues.filter(c=>c.id !== id))
  }

  return (
    <div style={{border:'1px solid #ddd', padding:12, marginBottom:8}}>
      <h3>Admin</h3>
      <form onSubmit={doUpload}>
        <input type="file" onChange={e=>setFile(e.target.files[0])} />
        <input placeholder="title" value={title} onChange={e=>setTitle(e.target.value)} />
        <div style={{marginTop:8}}><button>Upload</button></div>
      </form>
      <div style={{marginTop:12}}>
        <h4>Pending clues</h4>
        {clues.length===0 ? <div>No pending clues</div> : (
          <ul>{clues.map(c=>(
            <li key={c.id}>
              <strong>#{c.id}</strong> image:{c.image_id} by {c.contributor} — {c.text}
              <button style={{marginLeft:8}} onClick={()=>accept(c.id)}>Accept</button>
            </li>
          ))}</ul>
        )}
        <div style={{marginTop:8}}><button onClick={reloadClues}>Refresh</button></div>
      </div>
    </div>
  )
}

function Scoreboard() {
  const [rows, setRows] = useState([])
  useEffect(()=>{ api.scoreboard().then(r=>{ if(r.ok) setRows(r.body) }) },[])
  return (
    <div style={{border:'1px solid #ddd', padding:12}}>
      <h3>Scoreboard</h3>
      <ol>{rows.map(r=> <li key={r.name}>{r.name} — {r.score}</li>)}</ol>
    </div>
  )
}

/* --- App wrapper --- */
export default function App(){
  const [user, setUser] = useState(null)
  const [view, setView] = useState('images')
  const [selectedImageId, setSelectedImageId] = useState(null)

  useEffect(()=>{
    const token = localStorage.getItem('token')
    if (token){
      const u = parseJwt(token)
      if (u) setUser(u)
    }
  },[])

  function onLogout(){ clearToken(); setUser(null) }

  return (
    <div style={{padding:20, fontFamily:'system-ui,Arial'}}>
      <Header user={user} onLogout={onLogout} />
      <nav style={{marginBottom:12}}>
        <button onClick={()=>{ setView('images'); setSelectedImageId(null) }}>Images</button>{' '}
        <button onClick={()=>setView('score')}>Scoreboard</button>{' '}
        <button onClick={()=>setView('admin')}>Admin</button>{' '}
        {!user && <button onClick={()=>setView('login')}>Login/Register</button>}
      </nav>

      <main style={{display:'grid', gridTemplateColumns:'1fr 360px', gap:16}}>
        <div>
          {view === 'images' && <ImageList onOpen={(id)=>{ setSelectedImageId(id); setView('detail') }} />}
          {view === 'detail' && selectedImageId && <ImageDetail id={selectedImageId} user={user} onBack={()=>setView('images')} />}
          {view === 'login' && <div><Login onLogin={(u)=>{ setUser(u); setView('images') }} /><Register /></div>}
          {view === 'score' && <Scoreboard />}
        </div>

        <aside>
          {view === 'admin' ? <AdminPanel token={localStorage.getItem('token')} /> : (
            <>
              {!user ? <div><Login onLogin={(u)=>{ setUser(u); setView('images') }} /></div> : <div style={{border:'1px solid #ddd', padding:12}}>Logged in as <strong>{user?.name}</strong></div>}
              <div style={{marginTop:12}}><Scoreboard /></div>
            </>
          )}
        </aside>
      </main>
    </div>
  )
}
