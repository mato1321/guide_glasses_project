#打開 http://localhost:8000/admin 就可以用網頁管理
from fastapi import APIRouter, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse
import os
import shutil
import cv2
import numpy as np
router = APIRouter()
DB_PATH = "face_database"

@router.get("/admin", response_class=HTMLResponse)
async def admin_page():
    people = []
    if os.path.exists(DB_PATH):
        for name in sorted(os.listdir(DB_PATH)):
            person_path = os.path.join(DB_PATH, name)
            if os.path.isdir(person_path):
                photos = [f for f in os.listdir(person_path)
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                people.append({"name": name, "photo_count": len(photos)})

    # 生成 HTML
    rows = ""
    for p in people:
        rows += f"""
        <tr>
            <td style="padding:12px; font-size:18px;">{p['name']}</td>
            <td style="padding:12px;">{p['photo_count']} 張照片</td>
            <td style="padding:12px;">
                <form method="post" action="/admin/rename" style="display:inline;">
                    <input type="hidden" name="old_name" value="{p['name']}">
                    <input type="text" name="new_name" placeholder="新名字" 
                        style="padding:6px; width:120px;">
                    <button type="submit" style="padding:6px 12px;">改名</button>
                </form>
                <form method="post" action="/admin/delete" style="display:inline; margin-left:8px;">
                    <input type="hidden" name="name" value="{p['name']}">
                    <button type="submit" style="padding:6px 12px; background:#ff4444; color:white; border:none; border-radius:4px;"
                            onclick="return confirm('確定要刪除 {p['name']} 嗎？')">
                        刪除
                    </button>
                </form>
            </td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>人臉資料庫管理</title>
        <style>
            body {{ font-family: Arial; padding: 20px; max-width: 800px; margin: 0 auto; }}
            h1 {{ color: #333; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th {{ background: #4CAF50; color: white; padding: 12px; text-align: left; }}
            tr:nth-child(even) {{ background: #f2f2f2; }}
            tr:hover {{ background: #ddd; }}
            .upload-box {{ background: #e8f5e9; padding: 20px; border-radius: 8px; margin: 20px 0; }}
            input, button {{ font-size: 14px; }}
            .success {{ background: #4CAF50; color: white; padding: 10px; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <h1>人臉資料庫管理</h1>
        <p>目前共 <strong>{len(people)}</strong> 人已註冊</p>

        <div class="upload-box">
            <h3>新增人臉</h3>
            <form method="post" action="/admin/add" enctype="multipart/form-data">
                <label>姓名：</label>
                <input type="text" name="name" required style="padding:8px; width:200px;">
                <br><br>
                <label>照片：</label>
                <input type="file" name="files" accept="image/*" multiple required>
                <br><br>
                <button type="submit" style="padding:10px 24px; background:#4CAF50; color:white; border:none; border-radius:4px; font-size:16px;">
                    上傳並註冊
                </button>
            </form>
        </div>

        <h3>已註冊的人</h3>
        <table>
            <tr>
                <th>姓名</th>
                <th>照片數量</th>
                <th>操作</th>
            </tr>
            {rows}
        </table>

        <hr>
        <form method="post" action="/admin/reload">
            <button type="submit" style="padding:10px 24px; background:#2196F3; color:white; border:none; border-radius:4px;">
                重新載入資料庫
            </button>
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@router.post("/admin/add", response_class=HTMLResponse)
async def add_person(name: str = Form(...), files: list[UploadFile] = File(...)):
    person_path = os.path.join(DB_PATH, name)
    os.makedirs(person_path, exist_ok=True)
    saved = 0
    for i, file in enumerate(files):
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is not None:
            filename = f"{name}_{i + 1}.jpg"
            cv2.imwrite(os.path.join(person_path, filename), image)
            saved += 1

    return HTMLResponse(content=f"""
        <html><body>
        <div style="padding:20px; font-family:Arial;">
            <h2>已新增 {name}（{saved} 張照片）</h2>
            <p>記得點「重新載入資料庫」讓伺服器讀取新資料</p>
            <a href="/admin">← 返回管理頁面</a>
        </div>
        </body></html>
    """)


@router.post("/admin/rename", response_class=HTMLResponse)
async def rename_person(old_name: str = Form(...), new_name: str = Form(...)):
    if not new_name.strip():
        return HTMLResponse(content="<h2>新名字不能為空</h2><a href='/admin'>返回</a>")

    old_path = os.path.join(DB_PATH, old_name)
    new_path = os.path.join(DB_PATH, new_name.strip())

    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        return HTMLResponse(content=f"""
            <html><body>
            <div style="padding:20px; font-family:Arial;">
                <h2>已將「{old_name}」改名為「{new_name.strip()}」</h2>
                <p>記得點「重新載入資料庫」</p>
                <a href="/admin">← 返回管理頁面</a>
            </div>
            </body></html>
        """)
    else:
        return HTMLResponse(content=f"<h2>找不到 {old_name}</h2><a href='/admin'>返回</a>")


@router.post("/admin/delete", response_class=HTMLResponse)
async def delete_person(name: str = Form(...)):
    person_path = os.path.join(DB_PATH, name)
    if os.path.exists(person_path):
        shutil.rmtree(person_path)
        return HTMLResponse(content=f"""
            <html><body>
            <div style="padding:20px; font-family:Arial;">
                <h2>已刪除「{name}」</h2>
                <p>記得點「重新載入資料庫」</p>
                <a href="/admin">← 返回管理頁面</a>
            </div>
            </body></html>
        """)
    else:
        return HTMLResponse(content=f"<h2>找不到 {name}</h2><a href='/admin'>返回</a>")


@router.post("/admin/reload", response_class=HTMLResponse)
async def reload_from_admin():
    # 這裡需要引用 engine，透過 main.py 的全域變數
    from main import engine
    engine.face_database.clear()
    engine.load_database()

    return HTMLResponse(content=f"""
        <html><body>
        <div style="padding:20px; font-family:Arial;">
            <h2>資料庫已重新載入</h2>
            <p>目前已註冊：{engine.get_registered_names()}</p>
            <a href="/admin">← 返回管理頁面</a>
        </div>
        </body></html>
    """)