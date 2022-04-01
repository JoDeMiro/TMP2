ReadMe.md

A saját szerverről leszettem néhány access.log fájlt és betettem egy a trace mappába. Hogy valós adatokon tudjam tesztelni a rendszer.

Ezekből a python tracemaker.py parancs segítségével lehet tesztelő adatokat generálni. Ez összefűzi a fájlokat. Kiolvassa belőle HTTP GET kérések időpontját. Ezeket elmenti a get_times.txt fájlba. Csinál egy másik fájlt is ahol a Datum/Idő formátum helyett Timestamp másodperc alapján kerülnek mentésre.

A python performancetest.py pedig a 127.0.0.1:8080 portra küldi ezeket a kéréseket a megadott sorrendben végrehajtva külön szálakon és elkapja a válaszidőket is.

Igy lehet meghajtani valós adatokkal egy kiszolgálót. JMeter ugyanis ilyen idő alapú visszajátszásra egyszerűen képtelen.

Azt, hogy milyen ütemben legyen végrehajtva a feladat a performacetest.py-ban lehet beállítani. Ahogy azt is, hogy mennyi adatot használjon fel illetve, hogy mennyire gyorsítsa a lejátszás - a kérések beküldésének - a sebességét.

Menet közben jöttem rá, hogy ha megvan az a get_timestamps_diff.txt file amit a tracemaker.py segítségével áll ellő, akkor lehet, hogy JMeterbe is be tudom olvastatni valahogy, hogy ebből a filéből küldözgesse be a kéréseket.
