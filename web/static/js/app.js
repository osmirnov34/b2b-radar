function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
}

function toggleSwitch(el) {
  el.classList.toggle('on');
}

function openModal(id) {
  document.getElementById(id)?.classList.remove('hidden');
}

function closeModal(id) {
  document.getElementById(id)?.classList.add('hidden');
}

function setActiveChip(el, groupSelector) {
  document.querySelectorAll(groupSelector).forEach((chip) => chip.classList.remove('active'));
  el.classList.add('active');
}

document.addEventListener('click', (e) => {
  if (e.target.matches('.modal-backdrop')) {
    e.target.classList.add('hidden');
  }
});
