import sqlite3

# データベースに接続
conn = sqlite3.connect('my_database.db')

# カーソルを取得
cursor = conn.cursor()

# 画像情報を保存するテーブルを作成
cursor.execute('''
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        caption TEXT
    )
''')

# 画像情報をデータベースに挿入
filename = 'uploaded_image.jpg'
caption = 'アップロードされた画像'
cursor.execute('INSERT INTO images (filename, caption) VALUES (?, ?)', (filename, caption))

# データベースの変更を保存
conn.commit()

# データベースから画像情報を取得
cursor.execute('SELECT * FROM images WHERE id = ?', (1,))
image_info = cursor.fetchone()
print(image_info)

# データベース接続をクローズ
conn.close()
