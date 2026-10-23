// Everything on the Life page lives here. Add a line, rebuild, done.

// Quotes for the carousel. `note` is shown under the author when the attribution needs care.
export const quotes = [
  { text: 'It is not that we have a short time to live, but that we waste a lot of it.', who: 'Seneca', note: 'On the Shortness of Life' },
  { text: 'The first principle is that you must not fool yourself, and you are the easiest person to fool.', who: 'Richard Feynman', note: 'Caltech commencement, 1974' },
  { text: 'In three words I can sum up everything I’ve learned about life: it goes on.', who: 'Robert Frost' },
  { text: 'Do what you can, with what you have, where you are.', who: 'Theodore Roosevelt' },
  { text: 'It does not matter how slowly you go as long as you do not stop.', who: 'Confucius', note: 'often attributed' },
  { text: 'The best way to predict the future is to create it.', who: 'Peter Drucker', note: 'often attributed' },
  { text: 'Life is what happens to us while we are making other plans.', who: 'Allen Saunders', note: '1957; later popularised by John Lennon' },
  { text: 'Hardships often prepare ordinary people for an extraordinary destiny.', who: 'The Voyage of the Dawn Treader', note: '2010 film, often credited to C. S. Lewis' },
  { text: 'Success is not the key to happiness. Happiness is the key to success.', who: 'Albert Schweitzer', note: 'often attributed' },
  { text: 'Believe you can and you’re halfway there.', who: 'Theodore Roosevelt', note: 'often attributed' },
  { text: 'Do not dwell in the past, do not dream of the future, concentrate the mind on the present moment.', who: 'often attributed to the Buddha' },
];

// Books. status: 'reading' | 'read'. `takeaway` is one honest sentence in your words.
// Entries with `example: true` only show while developing (npm run dev), never on the live site.
export const books = [
  { title: 'Letters from a Stoic', author: 'Seneca', status: 'reading', takeaway: 'Replace with your own line.', example: true },
  { title: 'Surely You’re Joking, Mr. Feynman!', author: 'Richard Feynman', status: 'read', takeaway: 'Replace with your own line.', example: true },
];

// Running. Totals are yours to update; `months` is km per month for the current year (Jan first).
export const running = {
  example: true, // set to false once these are your real numbers
  year: 2026,
  totalKm: 412,
  longestKm: 21.1,
  best5k: '24:30',
  best10k: '52:10',
  months: [22, 30, 41, 35, 48, 26, 18, 39, 52, 61, 40, 0],
  races: [{ date: '2026', name: 'Half marathon (example)', time: '1:58:00' }],
  note: 'Miles, and what they clear up.',
};

// Sketch gallery: file names in public/media, newest first.
export const sketches = [
  { file: 'sk_art_6.jpg', title: 'Portrait study', medium: 'graphite' },
  { file: 'sk_art_1.jpg', title: 'Tree', medium: 'graphite' },
  { file: 'sk_art_2.jpg', title: 'Eye', medium: 'coloured pencil' },
  { file: 'sk_art_4.jpg', title: 'Flowers in a vase', medium: 'colour' },
  { file: 'sk_art_3.jpg', title: 'Sunset over water', medium: 'colour' },
  { file: 'sk_art_5.jpg', title: 'Lake with birds', medium: 'colour' },
];
