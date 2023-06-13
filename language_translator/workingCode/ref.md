function generateRandomLetterOf100Words() {
  let alphabet = "abcdefghijklmnopqrstuvwxyz";
  let letter = alphabet[Math.floor(Math.random() * alphabet.length)];
  let words = [];

  for (let i = 0; i < 100; i++) {
    let wordLength = Math.floor(Math.random() * 10) + 1;
    let word = "";

    for (let j = 0; j < wordLength; j++) {
      word += letter;
    }

    words.push(word);
  }

  return words.join(" ");
}

console.log(generateRandomLetterOf100Words());