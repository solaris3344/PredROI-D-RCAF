# Türkçe kısa kullanım notu

Bu ZIP doğrudan GitHub deposu olarak yüklenebilecek şekilde düzenlenmiştir. Veri setleri, model ağırlıkları ve büyük çıktı dosyaları bilerek eklenmemiştir.

1. `requirements.txt` ile bağımlılıkları kurun.
2. Cheng manifest ve MAT klasörü yollarını `scripts/run_locked_drcaf.ps1` içinde düzenleyin.
3. Ana hasta-ayrık D-RCAF deneyini çalıştırın.
4. Fixed-0.5, slice-random ve multi-seed kontrolleri ayrı PowerShell dosyalarıyla çalıştırın.
5. Harici BDNeuro testi için `scripts/run_external.ps1` içindeki kök dizinleri güncelleyin.

Makaledeki temel yenilik leakage-safe, patient-disjoint, cross-fitted PredROI protokolüdür. D-RCAF ikincil bir füzyon uzantısıdır; kilitli Cheng sonuçlarında D-RCAF ile Fixed-0.5 aynı hasta doğruluğunu verdiği için adaptif füzyonun sabit füzyondan üstün olduğu iddia edilmemelidir.
