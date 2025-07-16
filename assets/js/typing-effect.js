document.addEventListener('DOMContentLoaded', () => {
  const el = document.getElementById('typing-text');
  if (!el) return;
  const text = el.getAttribute('data-text');
  let index = 0;
  el.innerHTML = '';
  function type() {
    if (index < text.length) {
      el.innerHTML += text.charAt(index);
      index++;
      setTimeout(type, 100); // adjust typing speed (in ms)
    }
  }
  type();
});
