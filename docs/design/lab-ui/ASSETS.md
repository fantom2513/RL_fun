# Шрифты и иконки прототипа

Все файлы лежат в репозитории; прототип и будущее приложение не обращаются к сети. Пакеты получены через npm (`npm install --ignore-scripts` во временном каталоге), из них скопированы только перечисленные файлы. Тексты лицензий — в `licenses/`.

| Пакет (npm) | Версия | Лицензия | Что скопировано | Размер |
|---|---|---|---|---|
| `@fontsource-variable/onest` | 5.3.1 | SIL OFL 1.1 | `onest-cyrillic-wght-normal.woff2`, `onest-latin-wght-normal.woff2` | 15,9 + 33,8 КБ |
| `@fontsource-variable/unbounded` | 5.3.0 | SIL OFL 1.1 | `unbounded-cyrillic-wght-normal.woff2`, `unbounded-latin-wght-normal.woff2` | 31,4 + 50,9 КБ |
| `@fontsource-variable/jetbrains-mono` | 5.3.0 | SIL OFL 1.1 | `jetbrains-mono-cyrillic-wght-normal.woff2`, `jetbrains-mono-latin-wght-normal.woff2` | 12,1 + 40,4 КБ |
| `lucide-static` | 1.53.0 | ISC | 78 иконок из `icons/*.svg`, собраны в спрайт `prototype/assets/icons.js` | 17,5 КБ |

- Шрифты переменные по весу (`wght`), подключены через `@font-face` с `unicode-range` (кириллица и базовая латиница), `font-display: swap`. Авторы: The Onest Project Authors (2021), The Unbounded Project Authors (2022), The JetBrains Mono Project Authors (2020).
- OFL разрешает встраивание и распространение вместе с программой; переименование при изменении файлов не требуется, пока файлы не изменены (они не изменены).
- ISC разрешает использование и распространение при сохранении уведомления об авторских правах (оно в `licenses/lucide-ISC.txt` и в комментарии `icons.js`).
- Данные трасс `prototype/assets/tracks.js` получены из `src/rl_fun/racing/tracks/*.json` этого репозитория.

Пересборка `icons.js` и `tracks.js`: `node docs/design/lab-ui/tools/build-assets.mjs <каталог node_modules с lucide-static>`. Node нужен только для пересборки; прототип открывается без него.

Единственная строка вида `http://…` в коде прототипа — идентификатор пространства имён SVG в `proto.js` (`createElementNS`); это не сетевой адрес (так же его исключает `tests/lab/test_web.py`).
