import io
from flask import Flask, render_template, request, jsonify
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# Muat model terbaik hasil training YOLOv8
model = YOLO('best.pt')

# Database informasi lengkap & rekomendasi tindakan tiap kelas
INFO_KEMATANGAN = {
    'mentah': {
        'judul': 'Buah Mentah (Unripe)',
        'warna': '#22c55e',  # Hijau
        'badge': 'Belum Siap Panen',
        'ciri_fisik': 'Buah berwarna hijau pekat menyeluruh, permukaan kulit masih keras, dan belum ada buah yang memberondol (lepas dari tandan).',
        'kandungan': 'Kadar rendemen minyak (CPO) sangat rendah (<12%), kandungan air dan getah masih sangat tinggi.',
        'rekomendasi': 'Jangan dipanen. Biarkan tandan tetap di pohon hingga warna kulit berubah menjadi oranye-kemerahan dan mulai memberondol.'
    },
    'mengkal': {
        'judul': 'Buah Mengkal (Underripe)',
        'warna': '#eab308',  # Kuning
        'badge': 'Hampir Matang',
        'ciri_fisik': 'Warna kulit buah mulai bergradasi dari hijau menjadi kekuningan atau kemerahan pada bagian ujung, baru terdapat sedikit brondolan alami.',
        'kandungan': 'Kadar minyak mulai terbentuk (sekitar 14%–18%), namun belum mencapai titik rendemen maksimal.',
        'rekomendasi': 'Tunda panen selama 3–7 hari ke depan hingga seluruh buah berubah warna menjadi oranye-merah cerah dan jumlah brondolan mencukupi.'
    },
    'matang': {
        'judul': 'Buah Matang Optimal (Ripe)',
        'warna': '#f97316',  # Oranye-Merah
        'badge': 'Siap Panen (Ideal)',
        'ciri_fisik': 'Warna kulit buah merah-oranye cerah merata, terdapat brondolan segar yang jatuh secara alami di piringan pohon.',
        'kandungan': 'Kadar rendemen minyak (CPO) maksimal (22%–26%) dengan kadar Asam Lemak Bebas (ALB/FFA) rendah (<3%).',
        'rekomendasi': 'Segera panen dan kirim ke Pabrik Kelapa Sawit (PKS) dalam waktu maksimal 24 jam untuk menjaga kualitas CPO terbaik.'
    },
    'lewat matang': {
        'judul': 'Buah Lewat Matang (Overripe)',
        'warna': '#ef4444',  # Merah Tua
        'badge': 'Segera Olah (Prioritas)',
        'ciri_fisik': 'Warna buah merah tua kusam, sebagian besar buah bagian luar sudah memberondol dan kelopak tandan mulai mengering.',
        'kandungan': 'Kadar minyak tinggi, namun kadar Asam Lemak Bebas (ALB/FFA) meningkat tajam (>5%) yang menurunkan mutu minyak.',
        'rekomendasi': 'Segera kutip seluruh brondolan dan prioritaskan pengolahan langsung di pabrik agar kenaikan ALB tidak semakin parah.'
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'Tidak ada file gambar yang diunggah'}), 400

    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    width, height = img.size

    # Jalankan deteksi menggunakan model YOLOv8
    # Turunkan conf ke 0.20 dan aktifkan augmentasi TTA agar objek kecil/banyak lebih mudah terbaca
    results = model.predict(img, conf=0.20, iou=0.45, augment=True)
    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id].lower()
            conf = round(float(box.conf[0]) * 100, 1)
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            info = INFO_KEMATANGAN.get(label, {
                'judul': label.title(),
                'warna': '#3b82f6',
                'badge': 'Terdeteksi',
                'ciri_fisik': '-',
                'kandungan': '-',
                'rekomendasi': '-'
            })

            detections.append({
                'label': label,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2],
                'info': info
            })

    # Urutkan deteksi berdasarkan tingkat keyakinan tertinggi
    detections.sort(key=lambda x: x['confidence'], reverse=True)

    return jsonify({
        'image_width': width,
        'image_height': height,
        'total_objects': len(detections),
        'detections': detections
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)