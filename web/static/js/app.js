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
