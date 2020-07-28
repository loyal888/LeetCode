package NEW_JOURNEY_IN_WORK.array;

/**
 * ！！！ 要注意待拼成单词的长度要小于chars
 */
public class LC_1160 {
    static final int SIZE = 26; // 数组大小为26-26个小写字母

    /**
     * 数组版本
     * @param words
     * @param chars
     * @return
     */
    public int countCharacters(String[] words, String chars) {
        /* 如果words为空或者长度为0
           如果chars为空或者长度为0
           直接返回0 */
        if (words == null || words.length == 0 || chars == null || chars.length() == 0)
            return 0;

        int spellOut = 0; // 可以拼出的字母的长度
        // boolean canSpell;    // 标识位，用来标识能否拼出
        int[] wordCounter;    // words字符计数器
        int[] charCounter = new int[SIZE];    // chars字符计数器

        /* 记录chars中每个字符出现的次数 */
        for (char c: chars.toCharArray()) {
            charCounter[c - 'a'] ++;
        }

        /* 记录words中的每个word的每个字符的出现的次数
           并与chars计数器比较 */
        loop:  // 标签，用来处理内部循环与外部循环之间的通信，如不使用，可以在循环使用一个boolean类型的变量来判断是否符合条件
        for (String w: words) {
            /* 如果w为空或者w长度为0或者w的长度>chars的长度，不参与统计 */
            if (w == null || w.length() == 0 || w.length() > chars.length())
                continue;

            wordCounter = new int[SIZE];
            // flag = true;
            for (char c: w.toCharArray()) {
                // 判断该字符是否在chars中出现
                if (charCounter[c - 'a'] == 0) continue loop; // canSpell = false;
                wordCounter[c - 'a'] ++;
            }
            /* 判断每个word的字符出现次数是否至少在chars中同样出现
               即word的字符出现次数是否<=chars出现的次数 */
            for (int i = 0; i < SIZE; i ++) {
                if (wordCounter[i] > charCounter[i])
                    continue loop;  // canSpell = false亦可；
            }
            //if (canSpell)
            spellOut += w.length();
        }
        return spellOut;
    }

    public static void main(String[] args) {
        new LC_1160().countCharacters(new String[]{"hello","world","leetcode"}, "welldonehoneyr");
    }
}
