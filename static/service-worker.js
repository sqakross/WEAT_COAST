// WCCR Technician PWA service worker
// Цель:
//  - максимально быстро грузить UI
//  - не показывать красную ошибку если Wi-Fi моргнул
//  - не ломать текущее поведение

const CACHE_NAME = "wccr-tech-cache-v1";

// Что кладём в кэш при установке SW.
// Здесь БЕЗОПАСНО оставить только глобальные вещи,
// которые точно есть в твоём проекте.
const URLS_TO_CACHE = [
  "/",                   // корень (redirect/dashboard)
  "/static/style.css",  // твой основной кастомный стиль
  // добавь сюда ещё, если хочешь предкешировать, например логотип
  // "/static/wccr-icon-192.png",
  // "/static/wccr-icon-512.png"
];

// Установка сервис-воркера: создаём кэш
self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(URLS_TO_CACHE))
      .catch(err => {
        console.warn("[SW install] cache addAll issue:", err);
      })
  );
});

// Стратегия на каждый запрос:
// 1. Пробуем пойти в сеть (чтобы данные актуальные).
// 2. Если сеть не отвечает → берём из кэша.
// 3. Если вообще ничего нет → если это навигация (HTML-страница),
//    даём кэшированный "/" чтобы не упасть с ошибкой.
self.addEventListener("fetch", event => {
  event.respondWith(
    fetch(event.request)
      .then(resp => resp)
      .catch(() => {
        return caches.match(event.request).then(cached => {
          if (cached) {
            return cached;
          }
          if (event.request.mode === "navigate") {
            // фоллбек — покажи хотя бы главную страницу
            return caches.match("/");
          }
        });
      })
  );
});

// Чистка старых кэшей при обновлении версии
self.addEventListener("activate", event => {
  const allow = [CACHE_NAME];
  event.waitUntil(
    caches.keys().then(keys => Promise.all(
      keys.map(key => {
        if (!allow.includes(key)) {
          return caches.delete(key);
        }
      })
    ))
  );
});
