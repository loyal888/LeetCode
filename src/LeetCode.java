import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;

import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


public class LeetCode {

    public static class ListNode {
        int val;
        @Nullable ListNode next = null;

        ListNode(int val) {
            this.val = val;
        }
    }

    public static class TreeNode {
        int val = 0;
        TreeNode left = null;
        TreeNode right = null;

        public TreeNode(int val) {
            this.val = val;

        }

    }


    public static int myAtoi(String str) {
        // 去除空格
        str = str.trim();
        // 保证为整数
        boolean matches = str.matches("[-+]?\\d+.*");
        int ans = 0;
        if (matches) {
            // 截取字符串
            String pattern = "[-+]?\\d+";
            Pattern p = Pattern.compile(pattern);
            Matcher matcher = p.matcher(str);
            if (matcher.find()) {
                str = matcher.group();
            }
            // 判断第一位是否为正负号
            char c = str.charAt(0);
            int pos = 0;
            int nag = 0;
            switch (c) {
                case '+':
                    pos = 1;
                    break;
                case '-':
                    nag = 1;
                    break;
                default:
                    break;
            }
            // 第一位为正数
            if (pos == 1) {
                for (int i = 1; i < str.length(); i++) {
                    Character num = str.charAt(i);
                    int res = Integer.valueOf(num.toString());
                    if (ans > Integer.MAX_VALUE / 10 || (ans == Integer.MAX_VALUE / 10 && res > 7)) {
                        return 2147483647;
                    }
                    ans = ans * 10 + res;
                }
            } else if (nag == 1) {
                for (int i = 1; i < str.length(); i++) {
                    Character num = str.charAt(i);
                    int res = Integer.valueOf(num.toString());
                    if (-ans < Integer.MIN_VALUE / 10 || (-ans == Integer.MIN_VALUE / 10 && res > 8)) {
                        return -2147483648;
                    }
                    ans = ans * 10 + res;
                }
                ans = ans * (-1);
            } else {
                for (int i = 0; i < str.length(); i++) {
                    Character num = str.charAt(i);
                    int res = Integer.valueOf(num.toString());
                    if (ans > Integer.MAX_VALUE / 10 || (ans == Integer.MAX_VALUE / 10 && res > 7)) {
                        return 2147483647;
                    }
                    ans = ans * 10 + res;
                }
            }
            return ans;
        }
        return 0;
    }

    public static String intToRoman(int num) {
        int[] nums = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};
        String[] romans = {"M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"};
        int index = 0;
        StringBuilder sb = new StringBuilder();
        while (index < nums.length) {

            while (num >= nums[index]) {
                sb.append(romans[index]);
                num = num - nums[index];
            }
            index++;
        }
        return sb.toString();

    }

    public static int romanToInt(String s) {
        Map<String, Integer> map = new HashMap<>();
        map.put("I", 1);
        map.put("IV", 4);
        map.put("V", 5);
        map.put("IX", 9);
        map.put("X", 10);
        map.put("XL", 40);
        map.put("L", 50);
        map.put("XC", 90);
        map.put("C", 100);
        map.put("CD", 400);
        map.put("D", 500);
        map.put("CM", 900);
        map.put("M", 1000);

        int ans = 0;
        for (int i = 0; i < s.length(); ) {
            if (i + 1 < s.length() && map.containsKey(s.substring(i, i + 2))) {
                ans += map.get(s.substring(i, i + 2));
                i += 2;
            } else {
                ans += map.get(s.substring(i, i + 1));
                i++;
            }
        }
        return ans;
    }

    /**
     * 14. 最长公共前缀
     * 标签：链表
     * 当字符串数组长度为 0 时则公共前缀为空，直接返回
     * 令最长公共前缀 ans 的值为第一个字符串，进行初始化
     * 遍历后面的字符串，依次将其与 ans 进行比较，两两找出公共前缀，最终结果即为最长公共前缀
     * 如果查找过程中出现了 ans 为空的情况，则公共前缀不存在直接返回
     * 时间复杂度：O(s)O(s)，s 为所有字符串的长度之和
     *
     * @param strs
     * @return
     */
    public static String longestCommonPrefix(String[] strs) {
        if (strs.length == 0)
            return "";
        String ans = strs[0];
        for (int i = 1; i < strs.length; i++) {
            int j = 0;
            for (; j < ans.length() && j < strs[i].length(); j++) {
                if (ans.charAt(j) != strs[i].charAt(j))
                    break;
            }
            ans = ans.substring(0, j);
            if (ans.equals(""))
                return ans;
        }
        return ans;
    }

    /**
     * 15. 三数之和
     *
     * @param nums
     * @return
     */
    @NotNull
    public static List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> ans = new ArrayList();
        int len = nums.length;
        if (nums == null || len < 3) return ans;
        Arrays.sort(nums); // 排序
        for (int i = 0; i < len; i++) {
            if (nums[i] > 0) break;
            // 如果当前数字大于0，则三数之和一定大于0，所以结束循环
            if (i > 0 && nums[i] == nums[i - 1]) continue; // 去重
            int L = i + 1;
            int R = len - 1;
            while (L < R) {
                int sum = nums[i] + nums[L] + nums[R];
                if (sum == 0) {
                    ans.add(Arrays.asList(nums[i], nums[L], nums[R]));
                    while (L < R && nums[L] == nums[L + 1]) L++; // 去重
                    while (L < R && nums[R] == nums[R - 1]) R--; // 去重
                    L++;
                    R--;
                } else if (sum < 0) L++;
                else if (sum > 0) R--;
            }
        }
        return ans;
    }

    /**
     * 16. 最接近的三数之和
     *
     * @param nums
     * @param target
     * @return
     */
    public int threeSumClosest(@Nullable int[] nums, int target) {
        if (nums == null || nums.length < 3) {
            return -1;
        }
        Arrays.sort(nums);
        int ans = nums[0] + nums[1] + nums[2];
        for (int i = 0; i < nums.length; i++) {
            int start = i + 1, end = nums.length - 1;
            while (start < end) {
                int sum = nums[i] + nums[start] + nums[end];
                if (Math.abs(target - sum) < Math.abs(target - ans)) {
                    ans = sum;
                }
                if (sum > target) {
                    end--;
                } else if (sum < target) {
                    start++;
                } else {
                    return ans;
                }
            }
        }
        return ans;
    }

    /**
     * 17. 电话号码的字母组合
     *
     * @param digits
     * @return
     */
    @NotNull
    public static List<String> letterCombinations(String digits) {
        LinkedList<String> ans = new LinkedList<String>();
        if (digits.isEmpty()) return ans;
        String[] mapping = new String[]{"0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        ans.add("");
        for (int i = 0; i < digits.length(); i++) {
//            int x = Character.getNumericValue(digits.charAt(i));
            int x = Integer.valueOf(digits.charAt(i) + "");
            while (ans.peek().length() == i) {
                String t = ans.remove();
                for (char s : mapping[x].toCharArray())
                    ans.add(t + s);
            }
        }
        return ans;
    }

    /**
     * 18. 四数之和
     *
     * @param nums
     * @param target
     * @return
     */
    public static List<List<Integer>> fourSum(@Nullable int[] nums, int target) {
        if (nums == null || nums.length < 4) {
            return new ArrayList<>();
        }
        Arrays.sort(nums);
        List<List<Integer>> ans = new ArrayList<>();
        for (int i = 0; i < nums.length; i++) {
            for (int j = i + 1; j < nums.length; j++) {
                int start = j + 1, end = nums.length - 1;

                while (start < end) {
                    int temp = nums[i] + nums[j] + nums[start] + nums[end];
                    if (target == temp) {
                        ArrayList<Integer> tList = new ArrayList<>();
                        tList.add(nums[i]);
                        tList.add(nums[j]);
                        tList.add(nums[start]);
                        tList.add(nums[end]);
                        ans.add(tList);
                        start++;
                        end--;
                    } else if (target > temp) {
                        start++;

                    } else {
                        end--;
                    }
                }

            }
        }
        return new ArrayList<>(new HashSet<>(ans));
    }

    /**
     * 19.删除链表的倒数第N个节点
     *
     * @param head
     * @param n
     * @return
     */
    @Nullable
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode p = new ListNode(0);
        p.next = head;
        ListNode first = p;
        ListNode second = p;
        // 提前移动 n+1步
        for (int i = 1; i <= n + 1; i++) {
            first = first.next;
        }
        while (first != null) {
            first = first.next;
            second = second.next;
        }
        second.next = second.next.next;
        return p.next;
    }

    /**
     * 20. 有效的括号
     *
     * @param s
     * @return
     */
    public static boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        HashMap<Character, Character> map = new HashMap<>();
        map.put(')', '(');
        map.put('}', '{');
        map.put(']', '[');
        for (int i = 0; i < s.length(); i++) {
            Character c = s.charAt(i);
            if (map.containsKey(c)) {
                Character value = stack.empty() ? '#' : stack.pop();
                if (!value.equals(map.get(c))) {
                    return false;
                }
            } else {
                stack.push(c);
            }
        }
        return stack.isEmpty();
    }

    /**
     * 21. 合并两个有序链表
     *
     * @param l1
     * @param l2
     * @return
     */
    @Nullable
    public static ListNode mergeTwoLists(@Nullable ListNode l1, @Nullable ListNode l2) {

        if (l1 == null) {
            return l2;
        }
        if (l2 == null) {
            return l1;
        }

        if (l1.val < l2.val) {
            l1.next = mergeTwoLists(l1.next, l2);
            return l1;
        } else {
            l2.next = mergeTwoLists(l1, l2.next);
            return l2;
        }
    }

    /**
     * 23. 合并K个排序链表
     *
     * @param lists
     * @return
     */
    @Nullable
    public static ListNode mergeKLists(@Nullable ListNode[] lists) {
        if (lists == null || lists.length == 0) {
            return null;
        }

        return merge(lists, 0, lists.length - 1);


    }

    @Nullable
    static ListNode merge(ListNode[] lists, int left, int right) {
        if (left == right) {
            return lists[left];
        }
        int mid = left + (right - left) / 2;
        ListNode l1 = merge(lists, left, mid);
        ListNode l2 = merge(lists, mid + 1, right);
        return mergeTwoLists(l1, l2);
    }

    /**
     * @param nums
     */
    public void nextPermutation(@NotNull int[] nums) {
        int i = nums.length - 2;
        while (i >= 0 && nums[i + 1] <= nums[i]) {
            i--;
        }
        if (i >= 0) {
            int j = nums.length - 1;
            while (j >= 0 && nums[j] <= nums[i]) {
                j--;
            }
            swap(nums, i, j);
        }
        reverse(nums, i + 1);
    }

    private void reverse(int[] nums, int start) {
        int i = start, j = nums.length - 1;
        while (i < j) {
            swap(nums, i, j);
            i++;
            j--;
        }
    }

    private void swap(int[] nums, int i, int j) {
//        int temp = nums[i];
//        nums[i] = nums[j];
//        nums[j] = temp;
//          a = a^b;
//          b = b^a;
//          a = a^b;
        nums[i] = nums[i] ^ nums[j];
        nums[j] = nums[j] ^ nums[i];
        nums[i] = nums[i] ^ nums[j];
    }


    /**
     * 32 最长有效括号
     *
     * @param s
     * @return
     */
    public static int longestValidParentheses(@Nullable String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }

        int ans = 0;
        Stack<Integer> stack = new Stack<>();
        stack.push(-1);
        for (int i = 0; i < s.length(); i++) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else {
                stack.pop();
                if (stack.empty()) {
                    stack.push(i);
                } else {
                    ans = Math.max(ans, i - stack.peek());
                }
            }
        }
        return ans;
    }

    /**
     * 33. 搜索旋转排序数组
     *
     * @param nums
     * @param target
     * @return
     */
    public int search(@NotNull int[] nums, int target) {
        int lo = 0, hi = nums.length - 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if ((nums[0] > target) ^ (nums[0] > nums[mid]) ^ (target > nums[mid]))
                lo = mid + 1;
            else
                hi = mid;
        }
        return lo == hi && nums[lo] == target ? lo : -1;
    }

    /**
     * 34. 在排序数组中查找元素的第一个和最后一个位置
     *
     * @param nums
     * @param target
     * @param left
     * @return
     */
    private static int extremeInsertionIndex(int[] nums, int target, boolean left) {
        int lo = 0;
        int hi = nums.length;

        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (nums[mid] > target || (left && target == nums[mid])) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }

        return lo;
    }

    @NotNull
    public static int[] searchRange(@NotNull int[] nums, int target) {
        int[] targetRange = {-1, -1};

        int leftIdx = extremeInsertionIndex(nums, target, true);

        // assert that `leftIdx` is within the array bounds and that `target`
        // is actually in `nums`.
        if (leftIdx == nums.length || nums[leftIdx] != target) {
            return targetRange;
        }

        targetRange[0] = leftIdx;
        targetRange[1] = extremeInsertionIndex(nums, target, false) - 1;

        return targetRange;
    }


    @NotNull
    private List<List<Integer>> res = new ArrayList<>();
    private int[] candidates;
    private int len;

    /**
     * 39. 组合总和
     * 注意：后面选取的数不能比前面选的数还要小
     *
     * @param residue
     * @param start
     * @param pre
     */
    private void findCombinationSum(int residue, int start, @NotNull Stack<Integer> pre) {
        if (residue == 0) {
            // Java 中可变对象是引用传递，因此需要将当前 path 里的值拷贝出来
            res.add(new ArrayList<>(pre));
            return;
        }
        // 优化添加的代码2：在循环的时候做判断，尽量避免系统栈的深度
        // residue - candidates[i] 表示下一轮的剩余，如果下一轮的剩余都小于 0 ，就没有必要进行后面的循环了
        // 这一点基于原始数组是排序数组的前提，因为如果计算后面的剩余，只会越来越小
        for (int i = start; i < len && residue - candidates[i] >= 0; i++) {
            pre.add(candidates[i]);
            // 【关键】因为元素可以重复使用，这里递归传递下去的是 i 而不是 i + 1
            findCombinationSum(residue - candidates[i], i, pre);
            pre.pop();
        }
    }

    @NotNull
    public List<List<Integer>> combinationSum(@NotNull int[] candidates, int target) {
        int len = candidates.length;
        if (len == 0) {
            return res;
        }
        // 优化添加的代码1：先对数组排序，可以提前终止判断
        Arrays.sort(candidates);
        this.len = len;
        this.candidates = candidates;
        findCombinationSum(target, 0, new Stack<>());
        return res;
    }

    /**
     * 42. 接雨水
     *
     * @param height
     * @return
     */
    public static int trap(int[] height) {
        int ans = 0;
        int size = height.length;
        for (int i = 1; i < size - 1; i++) {
            int max_left = 0, max_right = 0;
            for (int j = i; j >= 0; j--) {
                max_left = Math.max(max_left, height[j]);
            }
            for (int j = i; j < size; j++) {
                max_right = Math.max(max_right, height[j]);
            }
            ans += Math.min(max_left, max_right) - height[i];
        }
        return ans;
    }

    /**
     * 46. 全排列
     *
     * @param nums
     * @return
     */
    @NotNull
    public static List<List<Integer>> permute(int[] nums) {

        List<List<Integer>> res = new ArrayList<>();
        int[] visited = new int[nums.length];
        backtrack(res, nums, new ArrayList<Integer>(), visited);
        return res;

    }

    private static void backtrack(@NotNull List<List<Integer>> res, int[] nums, ArrayList<Integer> tmp, int[] visited) {
        if (tmp.size() == nums.length) {
            res.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = 0; i < nums.length; i++) {
            if (visited[i] == 1) continue;
            visited[i] = 1;
            tmp.add(nums[i]);
            backtrack(res, nums, tmp, visited);
            visited[i] = 0;
            tmp.remove(tmp.size() - 1);
        }
    }

    /**
     * 48. 旋转图像
     *
     * @param matrix
     */
    public void rotate(@NotNull int[][] matrix) {
        int n = matrix.length;
        // 转置矩阵
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                int temp = matrix[j][i];
                matrix[j][i] = matrix[i][j];
                matrix[i][j] = temp;
            }
        }
        // 反转本行
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n / 2; j++) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[i][n - j - 1];
                matrix[i][n - j - 1] = tmp;
            }
        }
    }

    /**
     * 49. 字母异位词分组
     *
     * @param strs
     * @return
     */
    public static List<List<String>> groupAnagrams(@Nullable String[] strs) {
        if (strs == null || strs.length == 0) return new ArrayList<List<String>>();
        Map<String, List<String>> map = new HashMap<>();
        for (String str : strs) {
            char[] tmp = str.toCharArray();
            Arrays.sort(tmp);
            String keyStr = String.valueOf(tmp);
            if (!map.containsKey(keyStr)) map.put(keyStr, new ArrayList<String>());
            map.get(keyStr).add(str);
        }
        return new ArrayList<>(map.values());

    }

    /**
     * 53. 最大子序和
     * 暴力法
     *
     * @param nums
     * @return
     */
    public static int maxSubArray(int[] nums) {
        int tmp = nums[0];
        int max = tmp;
        int len = nums.length;
        for (int i = 1; i < len; i++) {
            // 当当前序列加上此时的元素的值大于tmp的值，说明最大序列和可能出现在后续序列中，记录此时的最大值
            if (tmp + nums[i] > nums[i]) {
                max = Math.max(max, tmp + nums[i]);
                tmp = tmp + nums[i];
            } else {   // 当tmp(当前和)小于下一个元素时，当前最长序列到此为止。以该元素为起点继续找最大子序列,
                // 并记录此时的最大
                max = Math.max(Math.max(max, tmp), Math.max(tmp + nums[i], nums[i]));
                tmp = nums[i];
            }
        }
        return max;

    }

    /**
     * 53. 最大子序和
     *
     * @param nums
     * @return
     */
    public int maxSubArray1(@NotNull int[] nums) {
        int ans = nums[0];
        int sum = 0;
        for (int num : nums) {
            if (sum > 0) {
                sum += num;
            } else {
                sum = num;
            }
            ans = Math.max(ans, sum);
        }
        return ans;
    }

//    /**
//     * 55. 跳跃游戏
//     * @param position
//     * @param nums
//     * @return
//     */
//    @Deprecated
//    public boolean canJumpFromPosition(int position, int[] nums) {
//        if (position == nums.length - 1) {
//            return true;
//        }
//
//        int furthestJump = Math.min(position + nums[position], nums.length - 1);
//        for (int nextPosition = position + 1; nextPosition <= furthestJump; nextPosition++) {
//            if (canJumpFromPosition(nextPosition, nums)) {
//                return true;
//            }
//        }
//
//        return false;
//    }
//
//    public boolean canJump(int[] nums) {
//        return canJumpFromPosition(0, nums);
//    }


    /**
     * 55. 跳跃游戏
     *
     * @param nums
     * @return
     */
    public boolean canJump(@NotNull int[] nums) {
        memo = new Index[nums.length];
        // 初始化 memo 的所有元素为 UNKNOWN，
        for (int i = 0; i < memo.length; i++) {
            memo[i] = Index.UNKNOWN;
        }
        // 除了最后一个显然是 GOOD （自己一定可以跳到自己）
        memo[memo.length - 1] = Index.GOOD;
        return canJumpFromPosition(0, nums);
    }

    enum Index {GOOD, BAD, UNKNOWN}

    Index[] memo;

    public boolean canJumpFromPosition(int position, @NotNull int[] nums) {
        if (memo[position] != Index.UNKNOWN) {
            // 如果已知直接返回结果 True / False
            return memo[position] == Index.GOOD;
        }

        int furthestJump = Math.min(position + nums[position], nums.length - 1);
        for (int nextPosition = position + 1; nextPosition <= furthestJump; nextPosition++) {
            if (canJumpFromPosition(nextPosition, nums)) {
                memo[position] = Index.GOOD;
                return true;
            }
        }

        memo[position] = Index.BAD;
        return false;
    }

    /**
     * 56. 合并区间
     *
     * @param intervals
     * @return
     */
    @NotNull
    public int[][] merge(@Nullable int[][] intervals) {
        List<int[]> res = new ArrayList<>();
        if (intervals == null || intervals.length == 0)
            return res.toArray(new int[0][]);

        // Arrays.sort(intervals, (a, b) -> a[0] - b[0]);// a[0] - b[0]大于0就交换顺序
        // 根据二维数组第一个数字大小按每一行整体排序
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                // TODO Auto-generated method stub
                return o1[0] - o2[0];
            }
        });
        int i = 0;
        while (i < intervals.length) {
            int left = intervals[i][0];
            int right = intervals[i][1];
            // i不能到最后一行,所以要小于(数组的长度 - 1)
            // 判断所在行的right和下一行的left大小,对right重新进行赋最大值,之后再不断进行while循环判断
            while (i < intervals.length - 1 && right >= intervals[i + 1][0]) {
                i++;
                right = Math.max(right, intervals[i][1]);
            }
            res.add(new int[]{left, right});
            i++;
        }
        return res.toArray(new int[0][]);
    }

    /**
     * 62. 不同路径
     *
     * @param m
     * @param n
     * @return
     */
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 0; i < n; i++) dp[0][i] = 1;
        for (int i = 0; i < m; i++) dp[i][0] = 1;
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }

    public int minPathSum(@NotNull int[][] grid) {
        int M = grid[0].length;
        int N = grid.length;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {
                if (i == 0 && j == 0) {
                    continue;
                }
                if (i == 0 && j > 0) {
                    grid[0][j] += grid[0][j - 1];
                    continue;
                }
                if (j == 0 && i > 0) {
                    grid[i][0] += grid[i - 1][0];
                    continue;
                }
                grid[i][j] += Math.min(grid[i - 1][j], grid[i][j - 1]);
            }
        }
        return grid[N - 1][M - 1];
    }

    /**
     * 70. 爬楼梯
     *
     * @param n
     * @return
     */
    public int climbStairs(int n) {
        if (n == 1) {
            return 1;
        }
        if (n == 2) {
            return 2;
        }

        int[] dp = new int[n];
        dp[0] = 1;
        dp[1] = 2;

        for (int i = 2; i < n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n - 1];
    }

    /**
     * 72. 编辑距离
     *
     * @param word1
     * @param word2
     * @return
     */
    public int minDistance(@NotNull String word1, @NotNull String word2) {
        int n1 = word1.length();
        int n2 = word2.length();
        int[][] dp = new int[n1 + 1][n2 + 1];
        // 第一行
        for (int j = 1; j <= n2; j++) dp[0][j] = dp[0][j - 1] + 1;
        // 第一列
        for (int i = 1; i <= n1; i++) dp[i][0] = dp[i - 1][0] + 1;

        for (int i = 1; i <= n1; i++) {
            for (int j = 1; j <= n2; j++) {
                if (word1.charAt(i - 1) == word2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j - 1], dp[i][j - 1]), dp[i - 1][j]) + 1;
                }
            }
        }
        return dp[n1][n2];
    }

    /**
     * 75. 颜色分类
     */
    public void sortColors(@NotNull int[] nums) {
        int p0 = 0, curr = 0;
        int p2 = nums.length - 1;
        int tmp;
        while (curr <= p2) {
            if (nums[curr] == 0) {
                tmp = nums[p0];
                nums[p0++] = nums[curr];
                nums[curr++] = tmp;
            } else if (nums[curr] == 2) {
                tmp = nums[curr];
                nums[curr] = nums[p2];
                nums[p2--] = tmp;
            } else curr++;
        }
    }

    // 用来标记该节点是否遍历过 因为只允许使用一次
    private boolean[][] marked;
    // 算出左上右下四个点坐标
    //          (x-1,y)
    // (x,y-1)   (x,y)  (x,y+1)
    //          (x+1,y)
    private int[][] directions = {{-1, 0}, {0, -1}, {0, 1}, {1, 0}};
    // 行数
    int m;
    // 列数
    int n;
    String word;
    private char[][] board;

    /**
     * 79. 单词搜索
     *
     * @param board
     * @param word
     * @return
     */
    public boolean exist(char[][] board, String word) {
        m = board.length;
        if (m == 0) {
            return false;
        }
        n = board[0].length;
        marked = new boolean[m][n];
        this.word = word;
        this.board = board;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (dfs(i, j, 0)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean inArea(int x, int y) {
        return x >= 0 && x < m && y >= 0 && y < n;
    }

    private boolean dfs(int i, int j, int start) {
        if (start == word.length() - 1) {
            return board[i][j] == word.charAt(start);
        }
        if (board[i][j] == word.charAt(start)) {
            marked[i][j] = true;
            for (int k = 0; k < 4; k++) {
                int newX = i + directions[k][0];
                int newY = j + directions[k][1];
                if (inArea(newX, newY) && !marked[newX][newY]) {
                    if (dfs(newX, newY, start + 1)) {
                        return true;
                    }
                }

            }
            marked[i][j] = false;
        }
        return false;
    }

    private void find(int[] nums, int begin, @NotNull List<Integer> pre) {
        // 没有显式的递归终止
        res.add(new ArrayList<>(pre));// 注意：Java 的引用传递机制，这里要 new 一下
        for (int i = begin; i < nums.length; i++) {
            pre.add(nums[i]);
            find(nums, i + 1, pre);
            pre.remove(pre.size() - 1);// 组合问题，状态在递归完成后要重置
        }
    }

    /**
     * 78. 子集
     *
     * @param nums
     * @return
     */
    public List<List<Integer>> subsets(@NotNull int[] nums) {
        int len = nums.length;
        res = new ArrayList<>();
        if (len == 0) {
            return res;
        }
        List<Integer> pre = new ArrayList<>();
        find(nums, 0, pre);
        return res;
    }


    /**
     * 94. 二叉树的中序遍历
     *
     * @param root
     * @return 时间复杂度：O(n)。
     * 空间复杂度：O(n))。
     */
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        TreeNode curr = root;
        while (curr != null || !stack.isEmpty()) {
            while (curr != null) {
                stack.push(curr);
                curr = curr.left;
            }
            curr = stack.pop();
            res.add(curr.val);
            curr = curr.right;
        }
        return res;
    }

    /**
     * 96. 不同的二叉搜索树
     *
     * @param n
     * @return
     */
    public int numTrees(int n) {
        int[] G = new int[n + 1];
        G[0] = 1;
        G[1] = 1;

        for (int i = 2; i <= n; ++i) {
            for (int j = 1; j <= i; ++j) {
                G[i] += G[j - 1] * G[i - j];
            }
        }
        return G[n];
    }

    /**
     * 98. 验证二叉搜索树
     *
     * @param root
     * @return
     */
    public boolean isValidBST(TreeNode root) {
        return helper(root, null, null);
    }

    public boolean helper(TreeNode node, Integer lower, Integer upper) {
        if (node == null) {
            return true;
        }
        int val = node.val;
        if (lower != null && val <= lower) {
            return false;
        }
        if (upper != null && val >= upper) {
            return false;
        }
        if (!helper(node.left, lower, val)) {
            return false;
        }
        if (!helper(node.right, val, upper)) {
            return false;
        }
        return true;
    }

    /**
     * 101. 对称二叉树
     *
     * @param root
     * @return
     */
    public boolean isSymmetric(TreeNode root) {
        return isMirror(root, root);
    }

    public boolean isMirror(TreeNode t1, TreeNode t2) {
        if (t1 == null && t2 == null) {
            return true;
        }
        if (t1 == null || t2 == null) {
            return false;
        }
        return (t1.val == t2.val) && isMirror(t1.left, t2.right) && isMirror(t2.left, t1.right);
    }

    List<List<Integer>> ans = new ArrayList<>();

    /**
     * 102. 二叉树的层次遍历
     *
     * @param root
     * @return
     */
    public List<List<Integer>> levelOrder(TreeNode root) {
        if (root == null) {
            return ans;
        }
        helper(root, 0);
        return ans;
    }

    private void helper(TreeNode node, int level) {
        if (ans.size() == level) {
            ans.add(new ArrayList<Integer>());
        }
        ans.get(level).add(node.val);
        if (node.left != null) {
            helper(node.left, level + 1);
        }
        if (node.right != null) {
            helper(node.right, level + 1);
        }

    }

    /**
     * 102. 二叉树的层次遍历
     *
     * @param root
     * @return
     */
    public List<List<Integer>> levelOrder1(TreeNode root) {
        List<List<Integer>> levels = new ArrayList<List<Integer>>();
        if (root == null) return levels;

        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.add(root);
        int level = 0;
        while (!queue.isEmpty()) {
            // start the current level
            levels.add(new ArrayList<Integer>());

            // number of elements in the current level
            int level_length = queue.size();
            for (int i = 0; i < level_length; ++i) {
                TreeNode node = queue.remove();

                // fulfill the current level
                levels.get(level).add(node.val);

                // add child nodes of the current level
                // in the queue for the next level
                if (node.left != null) queue.add(node.left);
                if (node.right != null) queue.add(node.right);
            }
            // go to next level
            level++;
        }
        return levels;
    }

    /**
     * 104. 二叉树的最大深度
     *
     * @param root
     * @return
     */
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    /**
     * 105. 从前序与中序遍历序列构造二叉树
     *
     * @param preorder
     * @param inorder
     * @return
     */
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int preLen = preorder.length;
        int inLen = inorder.length;
        if (preLen != inLen) {
            throw new RuntimeException("Incorrect input data.");
        }
        return buildTree(preorder, 0, preLen - 1, inorder, 0, inLen - 1);
    }


    /**
     * 使用数组 preorder 在索引区间 [preLeft, preRight] 中的所有元素
     * 和数组 inorder 在索引区间 [inLeft, inRight] 中的所有元素构造二叉树
     *
     * @param preorder 二叉树前序遍历结果
     * @param preLeft  二叉树前序遍历结果的左边界
     * @param preRight 二叉树前序遍历结果的右边界
     * @param inorder  二叉树后序遍历结果
     * @param inLeft   二叉树后序遍历结果的左边界
     * @param inRight  二叉树后序遍历结果的右边界
     * @return 二叉树的根结点
     */
    private TreeNode buildTree(int[] preorder, int preLeft, int preRight,
                               int[] inorder, int inLeft, int inRight) {
        // 因为是递归调用的方法，按照国际惯例，先写递归终止条件
        if (preLeft > preRight || inLeft > inRight) {
            return null;
        }
        // 先序遍历的起点元素很重要
        int pivot = preorder[preLeft];
        TreeNode root = new TreeNode(pivot);
        int pivotIndex = inLeft;
        // 严格上说还要做数组下标是否越界的判断 pivotIndex < inRight
        while (inorder[pivotIndex] != pivot) {
            pivotIndex++;
        }
        root.left = buildTree(preorder, preLeft + 1, pivotIndex - inLeft + preLeft,
                inorder, inLeft, pivotIndex - 1);
        root.right = buildTree(preorder, pivotIndex - inLeft + preLeft + 1, preRight,
                inorder, pivotIndex + 1, inRight);
        return root;
    }

    private TreeNode pre;

    /**
     * 114
     * 二叉树展开为链表
     *
     * @param root
     */
    public void flatten(TreeNode root) {
        if (root == null) {
            return;
        }
        if (pre != null) {
            pre.right = root;
            pre.left = null;
        }
        pre = root;
        flatten(root.left);
        flatten(root.right);
    }

    /**
     * 121. 买卖股票的最佳时机
     *
     * @param prices
     * @return
     */
    public int maxProfit(int[] prices) {
        int minprice = Integer.MAX_VALUE;
        int maxprofit = 0;
        for (int i = 0; i < prices.length; i++) {
            if (prices[i] < minprice)
                minprice = prices[i];
            else if (prices[i] - minprice > maxprofit)
                maxprofit = prices[i] - minprice;
        }
        return maxprofit;
    }

    /**
     * 136. 只出现一次的数字
     * XOR 满足交换律和结合律
     * a \oplus b \oplus a = (a \oplus a) \oplus b = 0 \oplus b = ba⊕b⊕a=(a⊕a)⊕b=0⊕b=b
     * 所以我们只需要将所有的数进行 XOR 操作，得到那个唯一的数字。
     *
     * @param nums
     * @return
     */
    public int singleNumber(int[] nums) {
        int ans = 0;
        for (int i = 0; i < nums.length; i++) {
            ans ^= nums[i];
        }
        return ans;
    }


    /**
     * 139. 单词拆分
     *
     * @param s
     * @param wordDict
     * @return
     */
    public boolean wordBreak(String s, List<String> wordDict) {
        if (wordDict == null || wordDict.size() == 0) return s.isEmpty();
        int n = s.length();
        boolean[] dp = new boolean[n + 1];
        dp[0] = true;
        for (int i = 1; i <= n; i++) {
            for (int j = i - 1; j >= 0; j--) {
                if (dp[j] && wordDict.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[n];
    }

    /**
     * 141. 环形链表
     *
     * @param head
     * @return
     */
    public boolean hasCycle(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow) {
                return true;
            }
        }
        return false;
    }

    private ListNode getIntersect(ListNode head) {
        ListNode tortoise = head;
        ListNode hare = head;
        while (hare != null && hare.next != null) {
            tortoise = tortoise.next;
            hare = hare.next.next;
            if (tortoise == hare) {
                return tortoise;
            }
        }

        return null;
    }

    /**
     * 142. 环形链表 II
     * <p>
     * 我们利用已知的条件：慢指针移动 1 步，快指针移动 2 步，来说明它们相遇在环的入口处。
     * （下面证明中的 tortoise 表示慢指针，hare 表示快指针）
     * 2 *distance(tortoise) = distance(hare)
     * 2(F+a) = F+a+b+a
     * 2F+2a = F+2a+b
     * F = b
     * 2⋅distance(tortoise)
     * 2(F+a)
     * 2F+2a
     * F=distance(hare)
     * =F+a+b+a
     * =F+2a+b
     * =b
     * 因为 F=b ，指针从 h 点出发和从链表的头出发，最后会遍历相同数目的节点后在环的入口处相遇。
     *
     * @param head
     * @return
     */
    public ListNode detectCycle(ListNode head) {
        if (head == null) {
            return null;
        }
        // 判断是否有环
        ListNode intersect = getIntersect(head);
        if (intersect == null) {
            return null;
        }
        ListNode ptr1 = head;
        ListNode ptr2 = intersect;
        while (ptr1 != ptr2) {
            ptr1 = ptr1.next;
            ptr2 = ptr2.next;
        }

        return ptr1;
    }

    /**
     * 148. 排序链表
     * 归并排序（递归法）
     *
     * @param head
     * @return
     */
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null)
            return head;
        ListNode fast = head.next, slow = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode tmp = slow.next;
        slow.next = null;
        ListNode left = sortList(head);
        ListNode right = sortList(tmp);
        ListNode h = new ListNode(0);
        ListNode res = h;
        while (left != null && right != null) {
            if (left.val < right.val) {
                h.next = left;
                left = left.next;
            } else {
                h.next = right;
                right = right.next;
            }
            h = h.next;
        }
        h.next = left != null ? left : right;
        return res.next;
    }

//归并排序（从底至顶直接合并）
//    class Solution:
//    def sortList(self, head: ListNode) -> ListNode:
//    h, length, intv = head, 0, 1
//            while h: h, length = h.next, length + 1
//    res = ListNode(0)
//    res.next = head
//        # merge the list in different intv.
//            while intv < length:
//    pre, h = res, res.next
//            while h:
//            # get the two merge head `h1`, `h2`
//    h1, i = h, intv
//                while i and h: h, i = h.next, i - 1
//            if i: break # no need to merge because the `h2` is None.
//    h2, i = h, intv
//                while i and h: h, i = h.next, i - 1
//    c1, c2 = intv, intv - i # the `c2`: length of `h2` can be small than the `intv`.
//            # merge the `h1` and `h2`.
//            while c1 and c2:
//            if h1.val < h2.val: pre.next, h1, c1 = h1, h1.next, c1 - 1
//            else: pre.next, h2, c2 = h2, h2.next, c2 - 1
//    pre = pre.next
//    pre.next = h1 if c1 else h2
//                while c1 > 0 or c2 > 0: pre, c1, c2 = pre.next, c1 - 1, c2 - 1
//    pre.next = h
//    intv *= 2
//    return res.next


    /**
     * 152. 乘积最大子序列
     *
     * @param nums
     * @return
     */
    public int maxProduct(int[] nums) {
        if (nums == null || nums.length == 0) return 0;
        int res = nums[0];
        int pre_max = nums[0];
        int pre_min = nums[0];
        for (int i = 1; i < nums.length; i++) {
            int cur_max = Math.max(Math.max(pre_max * nums[i], pre_min * nums[i]), nums[i]);
            int cur_min = Math.min(Math.min(pre_max * nums[i], pre_min * nums[i]), nums[i]);
            res = Math.max(res, cur_max);
            pre_max = cur_max;
            pre_min = cur_min;
        }
        return res;
    }

    /**
     * 155. 最小栈
     */
    class MinStack {
        private Stack<Integer> stack;
        private Stack<Integer> min_stack;

        public MinStack() {
            stack = new Stack<>();
            min_stack = new Stack<>();
        }

        public void push(int x) {
            stack.push(x);
            if (min_stack.isEmpty() || x <= min_stack.peek()) {
                min_stack.push(x);
            }
        }

        public void pop() {
            if (stack.pop().equals(min_stack.peek())) {
                min_stack.pop();
            }
        }

        public int top() {
            return stack.peek();
        }

        public int getMin() {
            return min_stack.peek();
        }
    }

    /**
     * 160. 相交链表
     *
     * @param headA
     * @param headB
     * @return
     */
    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        if (headA == null || headB == null) return null;
        ListNode pA = headA, pB = headB;
        while (pA != pB) {
            pA = pA == null ? headB : pA.next;
            pB = pB == null ? headA : pB.next;
        }
        return pA;
    }

    /**
     * 169. 求众数
     *
     * @param nums
     * @return
     */
    public int majorityElement(int[] nums) {
        int count = 0;
        Integer candidates = null;
        for (int num : nums) {
            if (count == 0) {
                candidates = num;
            }
            count += (num == candidates) ? 1 : -1;
        }
        return candidates;
    }

    /**
     * 198. 打家劫舍
     *
     * @param num
     * @return
     */
    public int rob(int[] num) {
        // 前一个值
        int preMax = 0;
        // 当前最大值
        int currMax = 0;
        for (int x : num) {
            int temp = currMax;
            currMax = Math.max(preMax + x, currMax);
            preMax = temp;
        }
        return currMax;
    }

    void dfs(char[][] grid, int r, int c) {
        int nr = grid.length;
        int nc = grid[0].length;

        if (r < 0 || c < 0 || r >= nr || c >= nc || grid[r][c] == '0') {
            return;
        }

        grid[r][c] = '0';
        dfs(grid, r - 1, c);
        dfs(grid, r + 1, c);
        dfs(grid, r, c - 1);
        dfs(grid, r, c + 1);
    }

// ================================================================================================

    /**
     * 200. 岛屿数量
     *
     * @param grid
     * @return
     */
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }

        int nr = grid.length;
        int nc = grid[0].length;
        int num_islands = 0;
        for (int r = 0; r < nr; ++r) {
            for (int c = 0; c < nc; ++c) {
                if (grid[r][c] == '1') {
                    ++num_islands;
                    dfs(grid, r, c);
                }
            }
        }

        return num_islands;
    }

// ================================================================================================

    /**
     * 200. 岛屿数量
     *
     * @param grid
     * @return
     */
    public int numIslands1(char[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }

        int nr = grid.length;
        int nc = grid[0].length;
        int num_islands = 0;

        for (int r = 0; r < nr; ++r) {
            for (int c = 0; c < nc; ++c) {
                if (grid[r][c] == '1') {
                    ++num_islands;
                    grid[r][c] = '0'; // mark as visited
                    Queue<Integer> neighbors = new LinkedList<>();
                    neighbors.add(r * nc + c);
                    while (!neighbors.isEmpty()) {
                        int id = neighbors.remove();
                        int row = id / nc;
                        int col = id % nc;
                        if (row - 1 >= 0 && grid[row - 1][col] == '1') {
                            neighbors.add((row - 1) * nc + col);
                            grid[row - 1][col] = '0';
                        }
                        if (row + 1 < nr && grid[row + 1][col] == '1') {
                            neighbors.add((row + 1) * nc + col);
                            grid[row + 1][col] = '0';
                        }
                        if (col - 1 >= 0 && grid[row][col - 1] == '1') {
                            neighbors.add(row * nc + col - 1);
                            grid[row][col - 1] = '0';
                        }
                        if (col + 1 < nc && grid[row][col + 1] == '1') {
                            neighbors.add(row * nc + col + 1);
                            grid[row][col + 1] = '0';
                        }
                    }
                }
            }
        }

        return num_islands;
    }

// ================================================================================================

    /**
     * 206. 反转链表
     *
     * @param head
     * @return
     */
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode p = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return p;
    }

    // =======================================分 割 线=========================================================

    /**
     * 栈排序
     *
     * @param stack
     */
    public static void sortStackByStack(Stack<Integer> stack) {
        //用于帮助排序的栈
        Stack<Integer> help = new Stack<Integer>();
        System.out.println(stack);
        while (!stack.isEmpty()) {
            //弹出一个元素
            int cur = stack.pop();
            //在帮助栈不为空并且弹出的元素小于帮助栈栈顶的元素就将帮助栈栈顶的元素压会元素数据栈
            while (!help.isEmpty() && help.peek() > cur) {
                stack.push(help.pop());
            }
            //将弹出的元素放到帮助栈中
            help.push(cur);
        }
        //现在的元素全部在帮助栈中从栈顶到栈底从大到小排列
        System.out.println(help);
        //将帮助栈的元素倒出到原数据栈
        while (!help.isEmpty()) {
            stack.push(help.pop());
        }
    }

    // =======================================分 割 线=========================================================

    /**
     * 674. 最长连续递增序列
     *
     * @param nums
     * @return
     */
    public int findLengthOfLCIS(int[] nums) {
        if (nums.length <= 1)
            return nums.length;
        int ans = 1;
        int count = 1;
        for (int i = 0; i < nums.length - 1; i++) {
            if (nums[i + 1] > nums[i]) {
                count++;
            } else {
                count = 1;
            }
            ans = count > ans ? count : ans;
        }
        return ans;
    }

    // =======================================分 割 线=========================================================

    /**
     * 队列Queue的链表实现（FIFO）
     */
    public class QueueImplByLinkList {
        private Node first; // 最早添加的结点连接
        private Node last;  // 最近添加的结点连接
        private int N;      // 元素数量

        // 定义结点的嵌套类
        private class Node {
            String string;
            Node next;
        }

        // 判断链表是否为空
        public boolean isEmpty() {
            return first == null;
        }

        // 统计链表结点数
        public int size() {
            return N;
        }

        // 向表尾添加元素
        public void enqueue(String s) {
            Node oldlast = last;
            last = new Node();
            last.string = s;
            last.next = null;
            if (isEmpty()) {
                first = last;
            } else {
                oldlast.next = last;
            }
            N++;
        }

        // 从表头删除元素
        public String dequeue() {
            String s = first.string;
            first = first.next;
            if (isEmpty()) {
                last = null;
            }
            N--;
            return s;
        }
    }

    // =======================================分 割 线=========================================================

    /**
     * 108. 将有序数组转换为二叉搜索树
     *
     * @param nums
     * @return
     */
    public TreeNode sortedArrayToBST(int[] nums) {
        Arrays.sort(nums);
        return sortedArrayToBST(nums, 0, nums.length);
    }

    private TreeNode sortedArrayToBST(int[] nums, int start, int end) {
        if (start == end) {
            return null;
        }
        int mid = (start + end) >>> 1;

        TreeNode root = new TreeNode(nums[mid]);
        root.left = sortedArrayToBST(nums, start, mid);
        root.right = sortedArrayToBST(nums, mid + 1, end);
        return root;
    }

    // =======================================分 割 线=========================================================

    /**
     * 662. 二叉树最大宽度
     *
     * @param root
     * @return
     */
    public static int widthOfBinaryTree(TreeNode root) {
        Queue<AnnotatedNode> queue = new LinkedList();
        queue.add(new AnnotatedNode(root, 0, 0));
        int curDepth = 0, left = 0, ans = 0;
        while (!queue.isEmpty()) {
            AnnotatedNode a = queue.poll();
            if (a.node != null) {
                queue.add(new AnnotatedNode(a.node.left, a.depth + 1, a.pos * 2));
                queue.add(new AnnotatedNode(a.node.right, a.depth + 1, a.pos * 2 + 1));
                if (curDepth != a.depth) {
                    curDepth = a.depth;
                    left = a.pos;
                }
                ans = Math.max(ans, a.pos - left + 1);
            }
        }
        return ans;
    }

    static class AnnotatedNode {
        TreeNode node;
        int depth, pos;

        AnnotatedNode(TreeNode n, int d, int p) {
            node = n;
            depth = d;
            pos = p;
        }
    }

    // =======================================分 割 线=========================================================

    /**
     * 一个n位数，现在可以删除其中任意k位，使得剩下的数最小
     *
     * @param a
     * @param k
     */
    static void findMinNums(int a, int k) {
        int n = 0;          //n指a有多少位
        int i, j, s = 1;

        n = ((int) Math.log10(a) + 1);

        int[] p = new int[n];
        j = a;
        for (i = n - 1; i >= 0; i--) {  //数组p中从0到n-1,分别为：5 6 3 1 7
            p[i] = j % 10;
            j /= 10;
        }

        for (i = 1; i <= k; i++) {  //控制删去的个数
            for (j = 0; j < n - i; j++) {
                if (p[j] < p[j + 1]) {  //如果递增，删除最后一个数
                    continue;
                } else {            //如果递减，删除第一个数
                    p[j] = -1;
                    break;
                }
            }
            for (j = 0; j < n - i; j++) {   //删掉的数的位置被赋值-1，现在把-1以后的数
                if (p[j] == -1) {     //都往前挪
                    p[j] = p[j + 1];
                    p[j + 1] = -1;
                }
            }
            for (i = 0; i < n - k; i++) {
                System.out.println(p[i]);
            }
        }
    }

    // =======================================分 割 线=========================================================

    /**
     * 两数相加(大数之和)
     *
     * @param l1
     * @param l2
     * @return
     */
    public static ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode pre = new ListNode(0);
        ListNode cur = pre;
        int carry = 0;
        while (l1 != null || l2 != null) {
            int x = l1 == null ? 0 : l1.val;
            int y = l2 == null ? 0 : l2.val;

            int sum = x + y + carry;
            carry = sum / 10;
            sum = sum % 10;
            cur.next = new ListNode(sum);
            cur = cur.next;
            if (l1 != null) {
                l1 = l1.next;
            }
            if (l2 != null) {
                l2 = l2.next;
            }
        }
        if (carry == 1) {
            cur.next = new ListNode(1);
        }
        return pre.next;
    }

    // =======================================分 割 线=========================================================

    /**
     * 232. 用栈实现队列
     */
    class MyQueue {
        Stack<Integer> s1 = new Stack<>();
        Stack<Integer> s2 = new Stack<>();


        /**
         * Initialize your data structure here.
         */
        public MyQueue() {

        }

        private int front;

        public void push(int x) {
            if (s1.empty()) {
                front = x;
            }
            while (!s1.isEmpty()) {
                s2.push(s1.pop());
            }
            s2.push(x);
            while (!s2.isEmpty()) {
                s1.push(s2.pop());
            }
        }


        /**
         * Removes the element from in front of queue and returns that element.
         */
        public int pop() {
            int ans = s1.pop();
            if (!s1.empty()) {
                front = s1.peek();
            }
            return ans;
        }

        /**
         * Get the front element.
         */
        public int peek() {
            return front;
        }

        /**
         * Returns whether the queue is empty.
         */
        public boolean empty() {
            return s1.isEmpty();
        }
    }
    // =======================================分 割 线=========================================================

    /**
     * 83. 删除排序链表中的重复元素
     *
     * @param head
     * @return
     */
    public ListNode deleteDuplicates(ListNode head) {
        ListNode current = head;
        while (current != null && current.next != null) {
            if (current.next.val == current.val) {
                current.next = current.next.next;
            } else {
                current = current.next;
            }
        }
        return head;
    }
    // =======================================分 割 线=========================================================

    /**
     * 82. 删除排序链表中的重复元素 II
     *
     * @param head
     * @return
     */
    public static ListNode deleteDuplicates2(ListNode head) {
        if (head == null) {
            return head;
        }
        if (head.next != null && head.val == head.next.val) {
            while (head.next != null && head.val == head.next.val) {
                head = head.next;
            }
            return deleteDuplicates2(head.next);
        } else {
            head.next = deleteDuplicates2(head.next);
        }
        return head;
    }
    // =======================================分 割 线=========================================================

    /**
     * @param x a double
     * @return the square root of x
     */
    public double sqrt(double x) {
        double left = 0.0;
        double right = x;
        double eps = 1e-10;

        // 因为小于1的小数的平方是小于它本身的
        // 即小于1的小数的平方根大于该数本身
        // 如果不使right = 1则无法用二分法找到该数
        if (right < 1.0) {
            right = 1.0;
        }

        while (right - left > eps) {
            // 二分浮点数 和二分整数不同
            // 一般都有一个精度的要求 譬如这题就是要求小数点后八位
            // 也就是只要我们二分的结果达到了这个精度的要求就可以
            // 所以 需要让 right 和 left 小于一个我们事先设定好的精度值 eps
            // 一般eps的设定1e-8,因为这题的要求是到1e-8,所以我把精度调到了1e-10
            // 和-8差两位，开完平方结果差一位，结果是9位小数，取8位不会出现误差
            // 最后 选择 left 或 right 作为一个结果即可
            double mid = left + (right - left) / 2;
            if (mid * mid < x) {
                left = mid;
            } else {
                right = mid;
            }
        }

        return left;
    }

    // =======================================分 割 线=========================================================

    /**
     * 一个无序有正有负数组，求乘积最大的三个数的乘积
     * 思路:定义五个数:一个最大,一个次大,一个第三大,一个最小,一个次小
     * 只要找出这五个数,问题就解决了
     *
     * @param arr
     * @return
     */
    private static long maxProduct(long[] arr) {
        //max1,max2,max3用于记录最大,次大,第三大的数
        long max1 = Integer.MIN_VALUE, max2 = Integer.MIN_VALUE, max3 = Integer.MIN_VALUE;
        //min1,min2用于记录最小,次小的数
        long min1 = Integer.MAX_VALUE, min2 = Integer.MAX_VALUE;
        for (long num : arr) {
            if (num > max1) {
                max3 = max2;
                max2 = max1;
                max1 = num;
            } else if (num > max2) {
                max3 = max2;
                max2 = num;
            } else if (num > max3) {
                max3 = num;
            }
            if (num < min1) {
                min2 = min1;
                min1 = num;
            } else if (num < min2) {
                min2 = num;
            }
        }
        return Math.max(max1 * max2 * max3, min1 * min2 * max1);
    }

    // =======================================分 割 线=========================================================

    /**
     * 125. 验证回文串
     *
     * @param head
     * @return
     */
    public boolean isPalindrome(ListNode head) {
        if (head == null || head.next == null) {
            return true;
        }
        //快慢指针找到链表的中点
        ListNode fast = head.next.next;
        ListNode slow = head.next;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }

        //翻转链表前半部分
        ListNode pre = null;
        ListNode next = null;
        while (head != slow) {
            next = head.next;
            head.next = pre;
            pre = head;
            head = next;
        }
        //如果是奇数个节点，去掉后半部分的第一个节点。

        if (fast != null) {
            slow = slow.next;
        }
        //回文校验
        while (pre != null) {
            if (pre.val != slow.val) {
                return false;
            }
            pre = pre.next;
            slow = slow.next;
        }

        return true;

    }
    // =======================================分 割 线=========================================================

    /**
     * 链表相邻元素交换
     *
     * @param head
     * @return
     */
    public static ListNode swapPairs(ListNode head) {
        if (head == null) {
            return null;
        }
        if (head.next == null) {
            return head;
        }

        ListNode temp = head.next;
        head.next = swapPairs(temp.next);
        temp.next = head;
        return temp;
    }

    // =======================================分 割 线=========================================================
    TreeNode ansNode;

    private boolean recurseTree(TreeNode currentNode, TreeNode p, TreeNode q) {

        // If reached the end of a branch, return false.
        if (currentNode == null) {
            return false;
        }

        // Left Recursion. If left recursion returns true, set left = 1 else 0
        int left = recurseTree(currentNode.left, p, q) ? 1 : 0;

        // Right Recursion
        int right = recurseTree(currentNode.right, p, q) ? 1 : 0;

        // If the current node is one of p or q
        int mid = (currentNode == p || currentNode == q) ? 1 : 0;


        // If any two of the flags left, right or mid become True
        if (mid + left + right >= 2) {
            ansNode = currentNode;
        }

        // Return true if any one of the three bool values is True.
        return (mid + left + right > 0);
    }

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        // Traverse the tree
        recurseTree(root, p, q);
        return ansNode;
    }
    // =======================================分 割 线=========================================================

    /**
     * 25. K 个一组翻转链表
     */
    public static ListNode reverseKGroup(ListNode head, int k) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;

        ListNode pre = dummy;
        ListNode end = dummy;

        while (end.next != null) {
            for (int i = 0; i < k && end != null; i++) {
                end = end.next;
            }
            if (end == null) {
                break;
            }
            ListNode start = pre.next;
            ListNode next = end.next;
            end.next = null;
            pre.next = reverse(start);
            start.next = next;
            pre = start;

            end = pre;
        }
        return dummy.next;
    }

    private static ListNode reverse(ListNode head) {
        ListNode pre = null;
        ListNode curr = head;
        while (curr != null) {
            ListNode next = curr.next;
            curr.next = pre;
            pre = curr;
            curr = next;
        }
        return pre;
    }

    // =======================================分 割 线=========================================================
    static class SUMNUMBERS {
        private int sum = 0;

        /**
         * 129. 求根到叶子节点数字之和
         *
         * @param root
         * @return
         */
        public int sumNumbers(TreeNode root) {
            if (root == null) {
                return sum;
            }
            helper(root, 0);
            return sum;
        }

        private void helper(TreeNode node, Integer father) {
            if (node == null) {
                return;
            }
            int current = father * 10 + node.val;
            if (node.left == null && node.right == null) {
                sum += current;
                return;
            }
            helper(node.left, current);
            helper(node.right, current);
        }
    }
    // =======================================分 割 线=========================================================

    /**
     * 257. 二叉树的所有路径
     *
     * @param root
     * @return
     */
    public List<String> binaryTreePaths(TreeNode root) {
        LinkedList<String> paths = new LinkedList();
        construct_paths(root, "", paths);
        return paths;
    }

    public void construct_paths(TreeNode root, String path, LinkedList<String> paths) {
        if (root != null) {
            path += Integer.toString(root.val);
            // 当前节点是叶子节点
            if ((root.left == null) && (root.right == null)) {
                // 把路径加入到答案中
                paths.add(path);
            } else { // 当前节点不是叶子节点，继续递归遍历
                path += "->";
                construct_paths(root.left, path, paths);
                construct_paths(root.right, path, paths);
            }
        }
    }
    // =======================================分 割 线=========================================================

    /**
     * 1013. 将数组分成和相等的三个部分
     *
     * @param nums
     * @return
     */
    public boolean canThreePartsEqualSum(int[] nums) {
        // 求nums的平均值
        double sum = 0;
        for (int i : nums) {
            sum += i;
        }
        double ave = sum / 3.0;
        int count = 0;
        double s = 0;
        for (int i : nums) {
            s += i;
            if (s == ave) {
                count++;
                s = 0;
            }
        }
        return count == 3;
    }
    // =======================================分 割 线=========================================================

    /**
     * 26. 删除排序数组中的重复项
     */
    public static int removeDuplicates(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        int i = 0;
        for (int j = 1; j < nums.length; j++) {
            if (nums[j] != nums[i]) {
                i++;
                nums[i] = nums[j];
            }
        }
        return i + 1;
    }

    // =======================================分 割 线=========================================================
    static class TrieNode {

        // R links to node children
        private TrieNode[] links;

        private final int R = 26;

        private boolean isEnd;

        public TrieNode() {
            links = new TrieNode[R];
        }

        public boolean containsKey(char ch) {
            return links[ch - 'a'] != null;
        }

        public TrieNode get(char ch) {
            return links[ch - 'a'];
        }

        public void put(char ch, TrieNode node) {
            links[ch - 'a'] = node;
        }

        public void setEnd() {
            isEnd = true;
        }

        public boolean isEnd() {
            return isEnd;
        }
    }

    /**
     * 208. 实现 Trie (前缀树)
     */
    static class Trie {
        private TrieNode root;

        public Trie() {
            root = new TrieNode();
        }

        // Inserts a word into the trie.
        public void insert(String word) {
            TrieNode node = root;
            for (int i = 0; i < word.length(); i++) {
                char currentChar = word.charAt(i);
                if (!node.containsKey(currentChar)) {
                    node.put(currentChar, new TrieNode());
                }
                node = node.get(currentChar);
            }
            node.setEnd();
        }

        private TrieNode searchPrefix(String word) {
            TrieNode node = root;
            for (int i = 0; i < word.length(); i++) {
                char curLetter = word.charAt(i);
                if (node.containsKey(curLetter)) {
                    node = node.get(curLetter);
                } else {
                    return null;
                }
            }
            return node;
        }

        // Returns if the word is in the trie.
        public boolean search(String word) {
            TrieNode node = searchPrefix(word);
            return node != null && node.isEnd();
        }

        public boolean startsWith(String prefix) {
            TrieNode node = searchPrefix(prefix);
            return node != null;
        }
    }
    // =======================================分 割 线=========================================================

    /**
     * 215. 数组中的第K个最大元素
     *
     * @param nums
     * @param k
     * @return
     */
    public static int findKthLargest(int[] nums, int k) {
        // init heap 'the smallest element first'
        PriorityQueue<Integer> heap =
                new PriorityQueue<Integer>();

        // keep k largest elements in the heap
        for (int n : nums) {
            heap.add(n);
            if (heap.size() > k) {
                heap.poll();
            }
        }

        // output
        return heap.poll();
    }

    // =======================================分 割 线=========================================================

    /**
     * 221. 最大正方形
     *
     * @param matrix
     * @return
     */
    public static int maximalSquare(char[][] matrix) {
        int rows = matrix.length, cols = rows > 0 ? matrix[0].length : 0;
        int[][] dp = new int[rows + 1][cols + 1];
        int maxsqlen = 0;
        for (int i = 1; i <= rows; i++) {
            for (int j = 1; j <= cols; j++) {
                if (matrix[i - 1][j - 1] == '1') {
                    dp[i][j] = Math.min(Math.min(dp[i][j - 1], dp[i - 1][j]), dp[i - 1][j - 1]) + 1;
                    maxsqlen = Math.max(maxsqlen, dp[i][j]);
                }
            }
        }
        return maxsqlen * maxsqlen;
    }

    // =======================================分 割 线=========================================================

    /**
     * 226. 翻转二叉树
     *
     * @param root
     * @return
     */
    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return root;
        }
        TreeNode right = root.right;
        TreeNode left = root.left;
        invertTree(root.left);
        invertTree(root.right);
        root.right = left;
        root.left = right;
        return root;
    }
    // =======================================分 割 线=========================================================

    /**
     * 238. 除自身以外数组的乘积
     *
     * @param nums
     * @return
     */
    public static int[] productExceptSelf(int[] nums) {
        int[] res = new int[nums.length];
        int k = 1;
        for (int i = 0; i < res.length; i++) {
            res[i] = k;
            k = k * nums[i]; // 此时数组存储的是除去当前元素左边的元素乘积
        }
        k = 1;
        for (int i = res.length - 1; i >= 0; i--) {
            res[i] *= k; // k为该数右边的乘积。
            k *= nums[i]; // 此时数组等于左边的 * 该数右边的。
        }
        return res;
    }

    // =======================================分 割 线=========================================================

    /**
     * 240. 搜索二维矩阵 II
     *
     * @param matrix
     * @param target
     * @return
     */
    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0) {
            return false;
        }
        int raw = matrix.length;
        int col = matrix[0].length;
        for (int i = 0; i < raw; i++) {
            for (int j = col - 1; j >= 0; j--) {
                if (matrix[i][j] < target) {
                    break;
                }
                if (matrix[i][j] == target) {
                    return true;
                }
            }
        }
        return false;
    }
    // =======================================分 割 线=========================================================

    /**
     * 279. 完全平方数
     *
     * @param n
     * @return
     */
    public static int numSquares(int n) {
        // 默认初始化值都为0
        int[] dp = new int[n + 1];
        for (int i = 1; i <= n; i++) {
            // 最坏的情况就是每次+1
            dp[i] = i;
            for (int j = 1; i - j * j >= 0; j++) {
                // 动态转移方程
                dp[i] = Math.min(dp[i], dp[i - j * j] + 1);
            }
        }
        return dp[n];
    }
// =======================================分 割 线=========================================================

//    public static void moveZeroes(int[] nums) {
//        if (nums == null || nums.length < 2) {
//            return;
//        }
//        int p = 0;
//        int q = 0;
//        while (true) {
//            if (q == nums.length - 1 || p == nums.length - 1) {
//                return;
//            }
//            while (p < nums.length - 1 && nums[p] != 0) {
//                p++;
//            }
//            q = q+1;
//            while (q < nums.length - 1 && nums[q] == 0) {
//                q++;
//            }
//            if (p < q) {
//                // 交换值
//                int tmp = nums[p];
//                nums[p] = nums[q];
//                nums[q] = tmp;
//            }
//        }
//    }

    /**
     * 283. 移动零
     * 1.nums中，i指针用于存放非零元素
     * 2.j指针用于遍历寻找非零元素
     * （注：j指针找到一个非零元素后，方法nums[i]的位置，i++，用于下一个j指针找到的非零元素）
     * 3.j指针遍历完后，最后nums数组还有空位置，存放0即可
     *
     * @param nums
     */
    public void moveZeroes(int[] nums) {
        int i = 0;
        for (int j = 0; j < nums.length; j++) {
            if (nums[j] != 0) {
                if (i != j) {
                    nums[i] = nums[j];
                }
                i++;
            }
        }
        for (; i < nums.length; i++) {
            nums[i] = 0;
        }
    }

    // =======================================分 割 线=========================================================

    /**
     * 287. 寻找重复数
     *
     * @param nums
     * @return
     */
    public static int findDuplicate(int[] nums) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int len = nums.length;
        int[] dp = new int[len + 1];
        for (int i = 0; i < len; i++) {
            if (dp[nums[i]] == 0) {
                dp[nums[i]] = nums[i];
            } else {
                return dp[dp[nums[i]]];
            }
        }
        return -1;
    }
    // =======================================分 割 线=========================================================

    /**
     * 300. 最长上升子序列
     *
     * @param nums
     * @return
     */
    public static int lengthOfLIS(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        int[] dp = new int[nums.length];
        dp[0] = 1;
        int maxans = 1;
        for (int i = 1; i < dp.length; i++) {
            int maxval = 0;
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    maxval = Math.max(maxval, dp[j]);
                }
            }
            dp[i] = maxval + 1;
            maxans = Math.max(maxans, dp[i]);
        }
        return maxans;
    }

    /**
     * 27. 移除元素
     *
     * @param nums
     * @param val
     * @return
     */
    public int removeElement(int[] nums, int val) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int len = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == val) {
                continue;
            }
            nums[len] = nums[i];
            len++;
        }
        return len;
    }

    /**
     * 29. 两数相除
     * @param dividend
     * @param divisor
     * @return
     */
    public static int divide(int dividend, int divisor) {
        boolean sign = (dividend > 0) ^ (divisor > 0);
        int result = 0;
        if (dividend > 0) {
            dividend = -dividend;
        }
        if (divisor > 0) divisor = -divisor;
        while (dividend <= divisor) {
            int temp_result = -1;
            int temp_divisor = divisor;
            while (dividend <= (temp_divisor << 1)) {
                if (temp_divisor <= (Integer.MIN_VALUE >> 1)) break;
                temp_result = temp_result << 1;
                temp_divisor = temp_divisor << 1;
            }
            dividend = dividend - temp_divisor;
            result += temp_result;
        }
        if (!sign) {
            if (result <= Integer.MIN_VALUE) return Integer.MAX_VALUE;
            result = -result;
        }
        return result;
    }

    /**
     * 35. 搜索插入位置
     * @param nums
     * @param target
     * @return
     */
    public int searchInsert(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while(left <= right) {
            int mid = (left + right) / 2;
            if(nums[mid] == target) {
                return mid;
            } else if(nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return left;
    }

// ================================================================================
    /**
     * 38. 报数
     * @param n
     * @return
     */
    public static String countAndSay(int n) {
        String res = "1";

        for (int i = 1; i < n; i++) {
            res = next_time(res);
        }
        return res;


    }

    private static String next_time(String res) {
        int i = 0;
        int n = res.length();
        String ans = "";
        while (i < n) {
            int count = 1;
            while ((i < n - 1) && (res.charAt(i) == res.charAt(i + 1))) {
                i++;
                count++;
            }
            ans += (count + "" + res.charAt(i));
            i++;

        }
        return ans;
    }
    // ================================================================================
// residue 表示剩余，这个值一开始等于 target，基于题目中说明的"所有数字（包括目标数）都是正整数"这个条件
    // residue 在递归遍历中，只会越来越小
    private void findCombinationSum2(int[] candidates, int begin, int len, int residue, Stack<Integer> stack, List<List<Integer>> res) {
        if (residue == 0) {
            res.add(new ArrayList<>(stack));
            return;
        }
        for (int i = begin; i < len && residue - candidates[i] >= 0; i++) {
            // 这一步之所以能够生效，其前提是数组一定是排好序的，这样才能保证：
            // 在递归调用的统一深度（层）中，一个元素只使用一次。
            // 这一步剪枝操作基于 candidates 数组是排序数组的前提下
            if (i > begin && candidates[i] == candidates[i - 1]) {
                continue;
            }
            stack.add(candidates[i]);
            // 【关键】因为元素不可以重复使用，这里递归传递下去的是 i + 1 而不是 i
            findCombinationSum2(candidates, i + 1, len, residue - candidates[i], stack, res);
            stack.pop();
        }
    }

    /**
     * 40. 组合总和 II
     * @param candidates
     * @param target
     * @return
     */
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        int len = candidates.length;
        List<List<Integer>> res = new ArrayList<>();
        if (len == 0) {
            return res;
        }
        // 先将数组排序，这一步很关键
        Arrays.sort(candidates);
        findCombinationSum2(candidates, 0, len, target, new Stack<>(), res);
        return res;
    }
    // ================================================================================

    /**
     * 50. Pow(x, n)
     * @param x
     * @param n
     * @return
     */
        private double fastPow(double x, long n) {
            if (n == 0) {
                return 1.0;
            }
            double half = fastPow(x, n / 2);
            if (n % 2 == 0) {
                return half * half;
            } else {
                return half * half * x;
            }
        }
        public double myPow(double x, int n) {
            long N = n;
            if (N < 0) {
                x = 1 / x;
                N = -N;
            }

            return fastPow(x, N);
        }

    // =======================================分 割 线=========================================================
    public static void main(String[] args) {


        int[] candidates = {10, 1, 2, 7, 6, 1, 5};
        int target = 8;
        LeetCode solution = new LeetCode();
        solution.fastPow(2,2);
        List<List<Integer>> combinationSum2 = solution.combinationSum2(candidates, target);
        System.out.println(combinationSum2);


        countAndSay(5);
        int result = divide(7, -2);
        lengthOfLIS(new int[]{10, 9, 2, 5, 3, 7, 101, 18});

        int[] num = {1, 2, 3, 4};

        ListNode listNode = new ListNode(1);
        ListNode listNode1 = new ListNode(2);
        ListNode listNode2 = new ListNode(3);
        ListNode listNode3 = new ListNode(4);
        ListNode listNode4 = new ListNode(5);
        ListNode listNode5 = new ListNode(1);
        ListNode listNode6 = new ListNode(8);


        listNode.next = listNode1;
        listNode1.next = listNode2;
        listNode2.next = listNode3;
//        swapPairs(listNode);

//        deleteDuplicates2(listNode);
        listNode3.next = listNode4;
        listNode4.next = listNode6;
        listNode5.next = listNode6;
        reverseKGroup(listNode, 2);
//        addTwoNumbers(listNode, listNode3);


        int[] nums = {2, 3, -2, 0, 1, 6};
        char[][] grid = new char[][]{{'1', '1', '1'}, {'0', '1', '0'}, {'1', '0', '0'}, {'1', '0', '1'}};
        new LeetCode().reverseList(listNode);


        TreeNode treeNode = new TreeNode(3);
        TreeNode treeNode1 = new TreeNode(2);
        TreeNode treeNode2 = new TreeNode(4);
        TreeNode treeNode3 = new TreeNode(6);

        treeNode.left = treeNode1;
        treeNode.right = treeNode2;
        treeNode1.right = treeNode3;
    }
// ================================================================================================

// ================================================================================================

}
