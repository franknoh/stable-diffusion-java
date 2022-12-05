package org.franknoh;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.text.Normalizer;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
public class Tokenizer {
    private final JsonObject vocab;
    private final Map<List<String>, Integer> merges;
    private final int bos_token;
    private final int eos_token;
    private final int pad_token;
    private final Map<Integer, Character> bytes_table;

    private Map<Integer, Character> create_bytes_table() {
       Map<Integer, Character> table = new HashMap<>();
       int special_count = 0;
       Map<Byte, String> unicodeCategories = new HashMap<>();
       unicodeCategories.put(Character.COMBINING_SPACING_MARK, "Mc");
       unicodeCategories.put(Character.CONNECTOR_PUNCTUATION, "Pc");
       unicodeCategories.put(Character.CONTROL, "Cc");
       unicodeCategories.put(Character.CURRENCY_SYMBOL, "Sc");
       unicodeCategories.put(Character.DASH_PUNCTUATION, "Pd");
       unicodeCategories.put(Character.DECIMAL_DIGIT_NUMBER, "Nd");
       unicodeCategories.put(Character.ENCLOSING_MARK, "Me");
       unicodeCategories.put(Character.END_PUNCTUATION, "Pe");
       unicodeCategories.put(Character.FINAL_QUOTE_PUNCTUATION, "Pf");
       unicodeCategories.put(Character.FORMAT, "Cf");
       unicodeCategories.put(Character.INITIAL_QUOTE_PUNCTUATION, "Pi");
       unicodeCategories.put(Character.LETTER_NUMBER, "Nl");
       unicodeCategories.put(Character.LINE_SEPARATOR, "Zl");
       unicodeCategories.put(Character.LOWERCASE_LETTER, "Ll");
       unicodeCategories.put(Character.MATH_SYMBOL, "Sm");
       unicodeCategories.put(Character.MODIFIER_LETTER, "Lm");
       unicodeCategories.put(Character.MODIFIER_SYMBOL, "Sk");
       unicodeCategories.put(Character.NON_SPACING_MARK, "Mn");
       unicodeCategories.put(Character.OTHER_LETTER, "Lo");
       unicodeCategories.put(Character.OTHER_NUMBER, "No");
       unicodeCategories.put(Character.OTHER_PUNCTUATION, "Po");
       unicodeCategories.put(Character.OTHER_SYMBOL, "So");
       unicodeCategories.put(Character.PARAGRAPH_SEPARATOR, "Zp");
       unicodeCategories.put(Character.PRIVATE_USE, "Co");
       unicodeCategories.put(Character.SPACE_SEPARATOR, "Zs");
       unicodeCategories.put(Character.START_PUNCTUATION, "Ps");
       unicodeCategories.put(Character.SURROGATE, "Cs");
       unicodeCategories.put(Character.TITLECASE_LETTER, "Lt");
       unicodeCategories.put(Character.UNASSIGNED, "Cn");
       unicodeCategories.put(Character.UPPERCASE_LETTER, "Lu");
       for (int i = 0; i < 256; i++) {
           String category = unicodeCategories.get( (byte) (Character.getType(i) ) );
           if (category.charAt(0) != 'C' && category.charAt(0) != 'Z') {
               table.put(i, (char)i);
           } else {
                table.put(i, (char)(256 + special_count));
               special_count++;
           }
       }
       return table;
   }

   private List<List<String>> pairwise(List<String> seq) {
       List<List<String>> pairs = new ArrayList<>();
       for (String s : seq) {
           for (String t : seq) {
               if (!Objects.equals(s, t)) {
                   List<String> pair = new ArrayList<>();
                   pair.add(s);
                   pair.add(t);
                   pairs.add(pair);
               }
           }
       }
       return pairs;
   }

    public Tokenizer() {
        Gson gson = new Gson();
        try {
            this.vocab = gson.fromJson(Files.readString(Paths.get("src/main/resources/vocab.json")), JsonObject.class);
            this.bos_token = this.vocab.get("<|startoftext|>").getAsInt();
            this.eos_token = this.vocab.get("<|endoftext|>").getAsInt();
            this.pad_token = this.vocab.get("<|endoftext|>").getAsInt();
            String[] lines = Files.readString(Paths.get("src/main/resources/merges.txt")).split("\n");
            lines = Arrays.copyOfRange(lines, 1, lines.length);
            this.merges = new HashMap<>();
            for (int i=0; i < lines.length; i++) {
                String[] bigram = lines[i].split("");
                this.merges.put(List.of(bigram), i);
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        this.bytes_table = create_bytes_table();
    }

    public List<Integer> encode(String text) {
       text = Normalizer.normalize(text, Normalizer.Form.NFC);
       text = text.replaceAll("\\s+", " ");
       text = text.strip();
       text = text.toLowerCase();
       List<Integer> tokens = new ArrayList<>();
       tokens.add(this.bos_token);
       List<String> chunks = new ArrayList<String>();
        String chunk_pattern = "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+";
        Matcher m = Pattern.compile(chunk_pattern).matcher(text);
       while (m.find()) {
           chunks.add(m.group());
       }
       for(String chunk : chunks) {
           Character[] bytes = new Character[chunk.length()];
           for (int i = 0; i < chunk.length(); i++) {
               bytes[i] = this.bytes_table.get((int)chunk.charAt(i));
           }
           StringBuilder chunkBuilder = new StringBuilder();
           for (Character b : bytes) {
                chunkBuilder.append(b);
           }
           chunk = chunkBuilder.toString();
           List<String> words = bpe(chunk);
           for (String word : words) {
               if(this.vocab.get(word) != null) {
                   tokens.add(this.vocab.get(word).getAsInt());
               } else {
                   tokens.add(this.pad_token);
               }
           }
       }
       tokens.add(this.eos_token);

        int max_length = 77;
        if(tokens.size() > max_length){
              tokens = tokens.subList(0, max_length);
              tokens.set(tokens.size() - 1, this.eos_token);
       }else if(tokens.size() < max_length){
              tokens.addAll(Collections.nCopies(max_length - tokens.size(), this.pad_token));
       }

       return tokens;
    }

    public List<List<Integer>> encode_batch(List<String> texts) {
       List<List<Integer>> tokens = new ArrayList<>();
       for (String text : texts) {
           tokens.add(this.encode(text));
       }
       return tokens;
    }

    private List<String> bpe(String chunk) {
       List<String> words = new ArrayList<>();
       words.add(chunk);
       words.set(words.size() - 1, words.get(words.size() - 1) + "</w>");

       while (words.size() > 1) {
           List<List<String>> valid_pairs = new ArrayList<>();
           List<List<String>> pairs = pairwise(words);
           for (List<String> pair : pairs) {
               if (merges.containsKey(pair)) {
                   valid_pairs.add(pair);
               }
           }
           if (valid_pairs.size() == 0) {
               break;
           }

           List<String> best_pair = valid_pairs.get(0);
           for (List<String> pair : valid_pairs) {
               if (merges.get(pair) < merges.get(best_pair)) {
                   best_pair = pair;
               }
           }

           List<String> new_words = new ArrayList<>();
           for(int i=0; i < words.size(); i++) {
               if (Objects.equals(words.get(i), best_pair.get(1)) && new_words.size() > 0 && Objects.equals(new_words.get(new_words.size() - 1), best_pair.get(0))) {
                   new_words.set(new_words.size() - 1, best_pair.get(0) + best_pair.get(1));
               } else {
                   new_words.add(words.get(i));
               }
           }
           words = new_words;
       }
       return words;
    }
}
